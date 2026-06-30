function [im, lims] = gtfMakespectro_ver1(v24, ploton, varargin)
% Make 224x224x3 TF image for CNN (parula colormap)
% im is uint8 (0–255) by default to match JPEG training.
%
% Optional name/value pairs:
%   'OutFor'    : 'net'|'file'|'double'  (default 'net' → uint8)
%   'OutputSize': [H W] (default [224 224])
%   'Scale'     : 'robust'|'fixed'       (default 'robust')
%   'MinInput'  : value (for 'fixed')
%   'MaxInput'  : value (for 'fixed')
%   'CMap'      : Nx3 colormap (default parula(256))
%
% Returns:
%   im   : image for net / file (uint8) or double [0,1] if OutFor='double'
%   lims : [mininput maxinput] actually used

% ---- parse opts ----
ip = inputParser;
addParameter(ip,'OutFor','net');
addParameter(ip,'OutputSize',[224 224]);
addParameter(ip,'Scale','robust');
addParameter(ip,'MinInput',[]);
addParameter(ip,'MaxInput',[]);
persistent defaultCMap
if isempty(defaultCMap), defaultCMap = parula(256); end
addParameter(ip,'CMap',defaultCMap);   % IMPORTANT: 256 entries for uint8 indices
parse(ip,varargin{:});
opts = ip.Results;

% ---- constants ----
Fs   = 24000;
DurS = 1.1;
Nreq = round(Fs*DurS);
nFFT = 2^12;
W    = nFFT/2;
OL   = floor(W*0.80);
Fmin = 50; Fmax = 1400;

% ---- length enforce ----
v = v24(:);
Lv = numel(v);
if Lv < Nreq
    v = [v; zeros(Nreq - Lv, 1, 'like', v)];
elseif Lv > Nreq
    v = v(1:Nreq);
end

% ---- spectrogram power → dB ----
[~,F,~,P] = spectrogram(v, W, OL, nFFT, Fs);
k  = (F >= Fmin) & (F <= Fmax);
P  = P(k,:);
P  = pow2db(P + eps('like',P));

% ---- percussive removal (along freq) + background subtraction (across time) ----
winMed = max(3, round(size(P,1)/5));
if mod(winMed,2)==0, winMed = winMed+1; end
try
    Pmed = medfilt1(P, winMed, 'truncate', 1);
catch
    Pmed = medfilt1(P, winMed, [], 1);
end
P  = P - Pmed;
bg = movmean(median(P,2), winMed, 'Endpoints','shrink');
P  = P - bg;

% ---- flip low→bottom ----
P = flipud(P);

% ---- scale limits ----
switch lower(opts.Scale)
    case 'fixed'
        mininput = opts.MinInput;
        maxinput = opts.MaxInput;
        if isempty(mininput) || isempty(maxinput)
            error('When Scale="fixed", provide MinInput and MaxInput.');
        end
    otherwise
        sigma    = 1.4826 * mad(P(:),1);
        mininput = max(-15, -2*sigma);
        maxinput = min( 15,  2*sigma);
end
lims = [mininput maxinput];

% ---- map to colormap with 256 entries ----
% 1) rescale to [0,1] with the SAME limits you used during training
Iidx = im2uint8( rescale(P, 0, 1, 'InputMin', mininput, 'InputMax', maxinput) ); % 0..255
% 2) colormap → RGB (double [0,1])
Irgb = ind2rgb(Iidx, opts.CMap);
% 3) resize
Irgb = imresize(Irgb, opts.OutputSize);

% ---- choose output dtype/range to match your training/inference ----
switch lower(opts.OutFor)
    case {'net','file'}   % matches JPEG read via imageDatastore (uint8 0–255)
        im = im2uint8(Irgb);
    case 'double'         % double [0,1] if you really want that
        im = Irgb;
    otherwise
        error('OutFor must be ''net'', ''file'', or ''double''.');
end

% ---- optional quick look ----
if nargin>1 && ploton
    imagesc(Irgb); axis image off; title('TF image'); drawnow;
end
end
