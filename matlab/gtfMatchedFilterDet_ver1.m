function [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = ...
    gtfMatchedFilterDet_ver1(y, frange, s, sweep, thres, predffreq, predffreq_uncert, ploton)

% [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = ...
%   gtfMatchedFilterDet_ver1(y, frange, s, sweep, thres, predffreq, predffreq_uncert, ploton)
%
% Spectrogram cross-correlation detector for Gulf toadfish boatwhistle calls.

% Defaults / constants
W = 4096; fs = 24000; np2 = 13;
if isrow(y), y = y(:); end

% ---- DIAGNOSTIC ENTRY POINT (remove when done) ----
fprintf('  [det ENTRY] ny=%d  frange=[%.1f %.1f]  predffreq=%.1f\n', numel(y), frange(1), frange(2), predffreq);

% ---- Spectrogram
[~,F,T,Pxx] = spectrogram(y, W, floor(W*.8), 2^np2, fs);
nT = numel(T);
fprintf('  [det SPEC] nT=%d  nF_full=%d\n', nT, numel(F));
if nT == 0, fprintf('  [det] EARLY RETURN: nT==0\n'); [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = deal([]); return; end

% ---- Trim band generously: [frange(1)-3s, 2*frange(2)+3s]
band = (F >= (frange(1)-3*s)) & (F <= (2*frange(2)+3*s));
F    = F(band);
Pxx  = Pxx(band,:);
nF   = numel(F);
fprintf('  [det BAND] nF_trimmed=%d  bandHz=[%.1f %.1f]\n', nF, frange(1)-3*s, 2*frange(2)+3*s);
if nF == 0, fprintf('  [det] EARLY RETURN: nF==0\n'); [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = deal([]); return; end

% ---- Build F1 (inclusive edges) and F2
F1a = F(F >= frange(1) & F <= frange(2));
fprintf('  [det F1a] nkerns=%d\n', numel(F1a));
if isempty(F1a), fprintf('  [det] EARLY RETURN: F1a empty\n'); [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = deal([]); return; end
F1b    = F1a - sweep;
F1     = [F1a(:)'; F1b(:)'];    % 2 x nkerns
F2     = 2*F1;
nkerns = size(F1,2);

% ---- Percussive filtering + background subtraction (dB)
PxxMod = pow2db(Pxx + eps('like',Pxx));

winMed = max(3, round(size(PxxMod,1)/6));
if mod(winMed,2)==0, winMed = winMed + 1; end

% Display copy: background-subtracted only, before percussive filter
PxxMod_display = PxxMod - movmean(median(PxxMod,2), winMed, 'Endpoints','shrink');

% 1) Percussive removal along frequency (column-wise median)
try
    PxxMedC = medfilt1(PxxMod, winMed, 'truncate', 1);
catch
    PxxMedC = medfilt1(PxxMod, winMed, [], 1);  % older version matlab 
end
PxxMod = PxxMod - PxxMedC;

% 2) Background subtraction across time (per frequency bin)
bg     = movmean(median(PxxMod,2), winMed, 'Endpoints','shrink');
PxxMod = PxxMod - bg;

% ---- Kernel time grid
if numel(T) >= 2
    dt = T(2)-T(1);
else
    hop = W - floor(W*.8);
    dt  = hop/fs;
end
klengthsec = 0.334;
t     = 0:dt:klengthsec;
len_k = numel(t);
alpha = t / max(t(end), eps);      % 0..1

% ---- Allocate correlation mats
cmatrix   = zeros(nkerns, nT + len_k - 1, 'single');
u_cmatrix = zeros(nkerns, nT + len_k - 1, 'single');
l_cmatrix = zeros(nkerns, nT + len_k - 1, 'single');

% ---- Fast norms via conv of column energies
oneK = ones(1, len_k, 'single');

% ---- Loop over kernels
for j = 1:nkerns
    % Upper band (2*F1)
    fo2 = F2(1,j); f12 = F2(2,j);
    X2  = F - (fo2 + alpha*(f12-fo2));
    u_k = (1 - (X2.^2)/(s^2)) .* exp(-(X2.^2)/(2*s^2));

    % Lower band (F1)
    fo1 = F1(1,j); f11 = F1(2,j);
    X1  = F - (fo1 + alpha*(f11-fo1));
    l_k = (1 - (X1.^2)/(s^2)) .* exp(-(X1.^2)/(2*s^2));

    k = u_k + l_k;

    % Trim ±3s Hz around each band centre for efficiency
    u_trim = F > fo2-3*s & F < fo2+3*s;
    l_trim = F > fo1-3*s & F < fo1+3*s;
    u_P = PxxMod(u_trim,:);  uK = u_k(u_trim,:);
    l_P = PxxMod(l_trim,:);  lK = l_k(l_trim,:);

    % Correlate along time
    %  conv2(signal, flip(kernel)) ≡ xcorr2(signal, kernel)
    c   = conv2(PxxMod, rot90(k,2),  'full');  c   = c(size(k,1),  :);
    u_c = conv2(u_P,    rot90(uK,2), 'full');  u_c = u_c(size(uK,1),:);
    l_c = conv2(l_P,    rot90(lK,2), 'full');  l_c = l_c(size(lK,1),:);

    u_knorm = sqrt(sum(uK(:).^2)) + eps;
    l_knorm = sqrt(sum(lK(:).^2)) + eps;

    E_u = conv(sum(u_P.^2,1), oneK, 'full');
    E_l = conv(sum(l_P.^2,1), oneK, 'full');
    u_norm = sqrt(E_u) + eps;
    l_norm = sqrt(E_l) + eps;

    % Combined-band norm: restrict denominator to signal rows (l_trim ∪ u_trim).
    % PnormAll inflates the denominator with ~125 between-harmonic residual bins,
    % dropping cmatrix to ~0.23 while l/u scores reach ~0.60.
    PnormComb  = sqrt(E_l + E_u) + eps;
    kComb_norm = sqrt(sum(lK(:).^2) + sum(uK(:).^2)) + eps;

    cmatrix(j,:)   = single(c)   ./ (PnormComb  * kComb_norm);
    u_cmatrix(j,:) = single(u_c) ./ (u_norm     * u_knorm);
    l_cmatrix(j,:) = single(l_c) ./ (l_norm     * l_knorm);
end

% ---- Trim to nT columns
cmatrix   = cmatrix(:,   len_k:end);
u_cmatrix = u_cmatrix(:, len_k:end);
l_cmatrix = l_cmatrix(:, len_k:end);
if isempty(cmatrix), [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = deal([]); return; end

% ---- Frequency prior taper
predffreq_orig   = predffreq;
predffreq        = min(max(predffreq, frange(1)), frange(2));
predffreq_uncert = max(predffreq_uncert, 25);
if predffreq ~= predffreq_orig   % warn if clamped
    warning('gtfMatchedFilterDet_ver1:predffreqClamped', ...
        'predffreq %.1f Hz outside frange [%.1f %.1f] — clamped to %.1f Hz.', ...
        predffreq_orig, frange(1), frange(2), predffreq);
end
lowcut  = predffreq - predffreq_uncert;
highcut = predffreq + predffreq_uncert;

  % lowcut and highcut are predffreq ± predffreq_uncert — the flat-top region where weight = 1. Outside that, the weight
  % decays exponentially on both sides, then floors at 0.25:
  % 
  % weight
  %  1.0  |     _____________
  %       |    /             \
  %  0.5  |   /               \
  %       |  /                 \
  %  0.25 |__                   __________
  %       |
  % lowcut-59Hz  lowcut  highcut  highcut+59Hz

xprd = (1:20) * 2.94; % decay over 59Hz 
yprd = exp(-(1:20)/15); % 15 is scale for decay 
xw   = [-12000, fliplr(lowcut - xprd), lowcut, highcut, highcut + xprd, 12000];
yw   = [0.25,   fliplr(yprd),          1,      1,       yprd,           0.25];
w    = interp1(xw, yw, F1(1,:), 'linear', 'extrap');
w    = movmean(w, 5);

% ---- Pick best kernel per time — taper biases selection AND scales cout.
% w(iout) applied to cout only; u_cout / l_cout remain untapered for band gates.
[~, iout] = max(cmatrix .* w(:), [], 1);   % kernel selection: taper-weighted
if isempty(iout), [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = deal([]); return; end
iout   = double(iout);
idx    = sub2ind(size(cmatrix), iout, 1:size(cmatrix,2));
cout   = double(cmatrix(idx)) .* w(iout);  % taper-weighted: off-freq calls suppressed
u_cout = double(u_cmatrix(idx));
l_cout = double(l_cmatrix(idx));

% ---- DIAGNOSTIC — remove once working
fprintf('  [det] nF=%d  nT=%d  nkerns=%d  len_k=%d\n', nF, nT, nkerns, len_k);
fprintf('  [det] max(cmatrix)=%.4f  max(l_cmatrix)=%.4f  max(u_cmatrix)=%.4f\n', ...
    max(cmatrix(:)), max(l_cmatrix(:)), max(u_cmatrix(:)));
fprintf('  [det] max(cout)=%.4f  max(l_cout)=%.4f  max(u_cout)=%.4f\n', ...
    max(cout), max(l_cout), max(u_cout));
fprintf('  [det] predffreq=%.1f  frange=[%.1f %.1f]\n', predffreq, frange(1), frange(2));

% ---- Peak picking
% min_peak_dist: at least one kernel-length apart to prevent double-detections on one call
min_peak_dist = max(4, round(klengthsec / dt));

% Combined-score peaks: both harmonics must produce a strong joint pattern.
% Only these can become f1_calls (CNN candidates).
[~, LOCS_c] = findpeaks(cout,   'MinPeakHeight',    thres, ...
                                 'MinPeakDistance',   min_peak_dist, ...
                                 'MinPeakProminence', thres/5);

% Lower-band-only peaks: catches fundamental-only calls (fo_calls).
% These are NEVER f1_calls — merging them into LOCS_c before the call-type
% gate was allowing noise with incidentally high u_scores to reach the CNN.
[~, LOCS_l] = findpeaks(l_cout, 'MinPeakHeight',    thres, ...
                                 'MinPeakDistance',   min_peak_dist, ...
                                 'MinPeakProminence', thres/5);

% Remove LOCS_l entries that are already covered by a combined peak
if ~isempty(LOCS_l) && ~isempty(LOCS_c)
    D      = abs(LOCS_l(:) - LOCS_c(:)');
    LOCS_l = LOCS_l(all(D > 9, 2));
end

LOCS_all = sort([LOCS_c(:)', LOCS_l(:)']);
fprintf('  [det] nLOCS_c=%d  nLOCS_l=%d  nLOCS_all=%d  thres=%.3f\n', numel(LOCS_c), numel(LOCS_l), numel(LOCS_all), thres);
if isempty(LOCS_all), [det_time,det_freq1,det_score,l_detscore,u_detscore,fo_calls,f1_calls] = deal([]); return; end

% ---- Score arrays
scores   = cout(LOCS_all).';
u_scores = u_cout(LOCS_all).';
l_scores = l_cout(LOCS_all).';
times    = T(LOCS_all).';
freqs    = F1(1, iout(LOCS_all)).';

% ---- Call-type logic
% f1_calls: MUST originate from a combined-score peak (cout > thres by findpeaks)
%           AND both individual band scores must also exceed thres.
%           LOCS_l-only entries cannot be f1_calls regardless of band scores.
tmp = ismember(LOCS_all, LOCS_c);
from_combined = tmp(:);
f1_calls = from_combined & l_scores > thres & u_scores > thres * 0.8;
fo_calls = l_scores > thres & ~f1_calls;

% ---- Final prune: keep rows with a detection
mask   = fo_calls | f1_calls;
locs_m = LOCS_all(mask);

det_time   = times(mask);
det_freq1  = freqs(mask);
det_score  = scores(mask);
u_detscore = u_scores(mask);
l_detscore = l_scores(mask);
fo_calls   = fo_calls(mask);
f1_calls   = f1_calls(mask);

% ---- Optional plots (#8: logical ploton)
if ploton
    figure; colormap('parula');

    ax(1) = subplot(5,1,1);
    imagesc(T, F, PxxMod_display); axis xy; clim([-35,35]);
    title('spectrogram'); ylabel('Hz');

    ax(2) = subplot(5,1,2);
    imagesc(T, F, PxxMod); axis xy; hold on; clim([-25,25]);
    title('percussive-filtered spectrogram with detections'); ylabel('Hz');
    plot(det_time(fo_calls), det_freq1(fo_calls), '+r', 'MarkerSize',4,'LineWidth',1);
    plot(det_time(f1_calls), det_freq1(f1_calls), 'ok', 'MarkerSize',4,'LineWidth',1);

    ax(3) = subplot(5,1,3); hold on;
    plot(T, cout, 'r'); plot(T, l_cout, 'g.'); plot(T, u_cout, 'b');
    if ~isempty(locs_m), plot(T(locs_m), det_score, 'or'); end
    grid on; ylim([0.1, 0.6]); ylabel('det score');

    ax(4) = subplot(5,1,4);
    imagesc(T, F1(1,:), cmatrix); axis xy;
    title('correlation with kernels (weighted)'); ylabel('F0 of kernel'); clim([-0.5,0.5]);

    ax(5) = subplot(5,1,5); hold on;
    for jj = 1:numel(locs_m)
        plot(T(locs_m(jj)) + cmatrix(:,locs_m(jj)), F1(1,:));
    end
    title('correlation vs kernel freq for candidates'); xlabel('time'); ylabel('F0 of kernel');
    linkaxes(ax,'x');
end
end

