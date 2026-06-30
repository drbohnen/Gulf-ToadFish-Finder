function [Bcount,Ocount,Btable,Otable,Dtable] = gtfDeepCatalogger_ver1( ...
    v24, site, segN, fileN, rootdir, classifier, Frange, predffreq, predffreq_uncert, ...
    fstartdatetime, NSEC, varargin)

% Per-segment classifier & detector — fine-tuned network version.
% classifier must be a struct with fields:
%   .net     — fine-tuned DAGNetwork (ResNet50 head replaced for 2 classes)
%   .bestThr — decision threshold on P(bwhistle)
%   .posIdx  — column index of 'bwhistle' in the network's softmax output
%
% Options:
%   'SaveImages'     (true|false, default false)
%   'KeepImages'     ('all'|'bwhistle'|'other'|'none', default 'all')
%   'UseParInternal' (true|false, default false)

%% --- Parse options
p = inputParser;
p.addParameter('SaveImages', false, @(x)islogical(x)&&isscalar(x));
p.addParameter('KeepImages','all', @(s)ischar(s)||isstring(s));
p.addParameter('UseParInternal', false, @(x)islogical(x)&&isscalar(x));
p.parse(varargin{:});
SaveImages     = p.Results.SaveImages;
KeepImages     = lower(string(p.Results.KeepImages));
UseParInternal = p.Results.UseParInternal;

minute_start = (str2double(segN)-1)*(NSEC/60);

%% --- Detector params
s=10; sweep=6; ploton=0; thres=0.45;
fs=24000;
winLen   = round(1.1*fs);
half1Win = round(0.4*fs);
half2Win = winLen - half1Win;

%% --- Run matched-filter detector
[dettimesMF,ffreq,ccscore,l_detscore,u_detscore,fo_calls,f1_calls] = ...
    gtfMatchedFilterDet_ver1(v24,Frange,s,sweep,thres,predffreq,predffreq_uncert,ploton);

Btable = table(); Otable = table(); Dtable = table();
Bcount = 0; Ocount = 0;

if isempty(dettimesMF)
    [Btable,Otable,Dtable] = makeTablesEmpty();
    return
end

%% --- Timing conversions & F1 subset
DetInSamples = round(fs*dettimesMF);
DetInSamples(DetInSamples+half2Win >= numel(v24)) = max(1, numel(v24)-(winLen+1));
DetTime  = DetInSamples./fs;

if ~isdatetime(fstartdatetime)
    fstartdatetime = datetime(fstartdatetime,'TimeZone','UTC');
elseif isempty(fstartdatetime.TimeZone)
    fstartdatetime.TimeZone = 'UTC';
end
DetTime2 = fstartdatetime + seconds(DetTime);

isF1 = f1_calls & (l_detscore > thres * 1.1) & (ccscore > thres * 0.8);
DetInSamples_f1 = DetInSamples(isF1);
DetTime_f1      = DetTime(isF1);
DetTime2_f1     = DetTime2(isF1);
ffreq_f1        = ffreq(isF1);
ccscore_f1      = ccscore(isF1);
l_detscore_f1   = l_detscore(isF1);
u_detscore_f1   = u_detscore(isF1);

%% --- D table (all detections)
Dtable = table( ...
    repmat(string(site), numel(dettimesMF), 1), ...
    ffreq(:), ccscore(:), l_detscore(:), u_detscore(:), ...
    fo_calls(:), f1_calls(:), ...
    DetTime(:), DetTime2(:), ...
    repmat(string(segN), numel(dettimesMF), 1), ...
    repmat(minute_start, numel(dettimesMF), 1), ...
    repmat(string(fileN), numel(dettimesMF), 1), ...
    'VariableNames', {'Site','ffreq','ccscore','l_ccscore','u_ccscore', ...
                      'fo_calls','f1_calls','rel_time','abs_time','segN','seg_min','file'});
Dtable = sortrows(Dtable,"abs_time");
Dtable.abs_time.TimeZone = 'UTC';

if isempty(DetInSamples_f1)
    return
end

%% --- Fine-tuned network inference (batched)
persistent netFine
if isempty(netFine)
    netFine = classifier.net;
end
posIdx  = classifier.posIdx;
bestThr = classifier.bestThr;
tgtSize = [224 224]; batch = 128;

nF1     = numel(DetInSamples_f1);
kStarts = 1:batch:nF1;
nBlocks = numel(kStarts);
blocks  = cell(1, nBlocks);

for b = 1:nBlocks
    k1 = kStarts(b);
    k2 = min(k1+batch-1, nF1);
    blocks{b} = batchPredict( ...
        v24, DetInSamples_f1, k1, k2, half1Win, winLen, tgtSize, ...
        netFine, posIdx, UseParInternal);
end

pd_b = zeros(nF1, 1, 'single');
for b = 1:nBlocks
    k1 = kStarts(b);
    k2 = min(k1+batch-1, nF1);
    pd_b(k1:k2) = blocks{b};
end

%% --- Classify using threshold
labs   = categorical(double(pd_b) >= bestThr, [0 1], {'other','bwhistle'});
bb     = find(labs == 'bwhistle');
oo     = find(labs == 'other');
Bcount = numel(bb);
Ocount = numel(oo);

%% --- Build B/O tables
if Bcount > 0
    Btable = table( ...
        repmat(string(site), Bcount, 1), ...
        categorical(repmat("bwhistle", Bcount,1)), ...
        ffreq_f1(bb), ccscore_f1(bb), l_detscore_f1(bb), u_detscore_f1(bb), ...
        ones(Bcount,1), ...
        double(pd_b(bb)), ...
        DetTime_f1(bb), DetTime2_f1(bb), ...
        repmat(string(segN), Bcount, 1), ...
        repmat(minute_start, Bcount, 1), ...
        repmat(string(fileN), Bcount, 1), ...
        'VariableNames', {'Site','CallID','ffreq','ccscore','l_ccscore','u_ccscore', ...
                          'f1_calls','pdscore','rel_time','abs_time','segN','seg_min','file'});
    Btable = sortrows(Btable,"abs_time");
    Btable.abs_time.TimeZone = 'UTC';
else
    Btable = makeEmptyLike();
end

if Ocount > 0
    Otable = table( ...
        repmat(string(site), Ocount, 1), ...
        categorical(repmat("other", Ocount,1)), ...
        ffreq_f1(oo), ccscore_f1(oo), l_detscore_f1(oo), u_detscore_f1(oo), ...
        ones(Ocount,1), ...
        double(pd_b(oo)), ...
        DetTime_f1(oo), DetTime2_f1(oo), ...
        repmat(string(segN), Ocount, 1), ...
        repmat(minute_start, Ocount, 1), ...
        repmat(string(fileN), Ocount, 1), ...
        'VariableNames', {'Site','CallID','ffreq','ccscore','l_ccscore','u_ccscore', ...
                          'f1_calls','pdscore','rel_time','abs_time','segN','seg_min','file'});
    Otable = sortrows(Otable,"abs_time");
    Otable.abs_time.TimeZone = 'UTC';
else
    Otable = makeEmptyLike();
end

%% --- Optional: write JPGs per segment
if SaveImages
    keyStem        = regexprep(fileN, '\.[^.]+$', '');
    out_directory  = fullfile(rootdir, keyStem);
    out_directoryB = fullfile(out_directory, 'bwhistle');
    out_directoryO = fullfile(out_directory, 'other');
    if ~exist(out_directory,'dir'),  mkdir(out_directory);  end
    if ~exist(out_directoryO,'dir'), mkdir(out_directoryO); end
    if ~exist(out_directoryB,'dir'), mkdir(out_directoryB); end

    writeChosenImages(1:numel(DetInSamples_f1), out_directoryO, site, segN, fileN, minute_start, ...
        DetInSamples_f1, DetTime_f1, v24, half1Win, winLen, [224 224]);

    if any(KeepImages == ["all","bwhistle"]) && ~isempty(Btable)
        for k = 1:height(Btable)
            base = strcat(fileN, sprintf('%03.0f',minute_start), '_', sprintf('%011.7f',Btable.rel_time(k)));
            base = strrep(base, '.wav', '_');
            src  = fullfile(out_directoryO, [base, '.jpg']);
            dst  = fullfile(out_directoryB, [base, '.jpg']);
            if exist(src,'file'), movefile(src, dst, 'f'); end
        end
    end

    if KeepImages == "bwhistle"
        try, rmdir(out_directoryO,'s'); catch, end
    end
end

end

%%%% helpers %%%%

function pd_b = batchPredict(v24, DetInSamples_f1, k1, k2, half1Win, winLen, tgtSize, netFine, posIdx, useParImg)
nb        = k2 - k1 + 1;
imgs      = zeros([tgtSize 3 nb], 'uint8');
det_slice = DetInSamples_f1(k1:k2);
nv        = numel(v24);
if useParImg
    parfor j = 1:nb
        winStart = max(1, det_slice(j) - half1Win);
        winEnd   = min(nv, winStart + winLen - 1);
        if (winEnd - winStart + 1) < winLen, winStart = max(1, winEnd - winLen + 1); end
        im = gtfMakespectro_ver1(v24(winStart:winEnd), 0, 'OutFor', 'net');
        if size(im,3) == 1, im = repmat(im,1,1,3); end
        imgs(:,:,:,j) = im;
    end
else
    for j = 1:nb
        winStart = max(1, det_slice(j) - half1Win);
        winEnd   = min(nv, winStart + winLen - 1);
        if (winEnd - winStart + 1) < winLen, winStart = max(1, winEnd - winLen + 1); end
        im = gtfMakespectro_ver1(v24(winStart:winEnd), 0, 'OutFor', 'net');
        if size(im,3) == 1, im = repmat(im,1,1,3); end
        imgs(:,:,:,j) = im;
    end
end
scores = predict(netFine, imgs, 'MiniBatchSize', 64);
pd_b   = single(scores(:, posIdx));
end

function [Btab,Otab,Dtab] = makeTablesEmpty()
Btab = makeEmptyLike();
Otab = makeEmptyLike();
Dtab = table(string.empty, [], [], [], [], [], [], [], datetime.empty, string.empty, [], string.empty, ...
    'VariableNames', {'Site','ffreq','ccscore','l_ccscore','u_ccscore','fo_calls','f1_calls','rel_time','abs_time','segN','seg_min','file'});
Dtab.abs_time.TimeZone = 'UTC';
end

function T = makeEmptyLike()
varNames = {'Site','CallID','ffreq','ccscore','l_ccscore','u_ccscore', ...
            'f1_calls','pdscore','rel_time','abs_time','segN','seg_min','file'};
varTypes = {'string','categorical','double','double','double','double', ...
            'double','double','double','datetime','string','double','string'};
T = table('Size',[0 numel(varNames)], 'VariableTypes',varTypes, 'VariableNames',varNames);
T.CallID = categorical(T.CallID, {'bwhistle','other'});
T.abs_time.TimeZone = 'UTC';
end

function writeChosenImages(idx, outDir, site, segN, fileN, minute_start, ...
        DetInSamples_f1, DetTime_f1, v24, half1Win, winLen, tgtSize)

if isempty(idx), return; end
if ~exist(outDir,'dir'), mkdir(outDir); end

batchW = 512;
H = tgtSize(1); W = tgtSize(2);

for p1 = 1:batchW:numel(idx)
    p2 = min(p1+batchW-1, numel(idx));
    nb = p2 - p1 + 1;

    imgs  = zeros(H, W, 3, nb, 'uint8');
    names = strings(nb,1);
    [~, fileBase] = fileparts(fileN);

    for j = 1:nb
        g = idx(p1 + j - 1);
        winStart = max(1, DetInSamples_f1(g) - half1Win);
        winEnd   = min(numel(v24), winStart + winLen - 1);
        if (winEnd - winStart + 1) < winLen
            winStart = max(1, winEnd - winLen + 1);
        end
        ydet = v24(winStart:winEnd);

        im = gtfMakespectro_ver1(ydet, 0, ...
            'OutFor','file', ...
            'Scale','robust', ...
            'CMap', parula(256));
        imgs(:,:,:,j) = im;

        names(j) = fullfile(outDir, sprintf('%s_%03.0f_%011.7f.jpg', ...
                        fileBase, minute_start, DetTime_f1(g)));
    end

    for j = 1:nb
        imwrite(imgs(:,:,:,j), names(j), 'jpg', 'Quality', 90);
    end
end
end
