%% ===================== Config =====================
clear;
datetime.setDefaultFormats('default','yyyy-MM-dd HH:mm:ss.SSS');

% Load fine-tuned classifier saved by gtfDeepToadFinetuneScript.m
% struct with fields: .net (DAGNetwork), .bestThr, .posIdx
tmp = load('gtfclassifier_tuned.mat');
classifier = tmp.classifier;

site = char('LM20180109');

% input setting and paths by site
if strcmp(site,'BK20180109')
DirIn  = 'K:\FLA\BK_20180109';
DirOut = '..\betaout\GTFdet_BK_20180109';
WT = readtable('K:\FLA\FLA_temperature\BK_gov-nps-ever-bkyf1_a830_1301_7798.csv');
WT(isnan(WT.wtemp_C),:) = [];
WT.UTC = datetime(WT.UTC,'InputFormat','yyyy-MM-dd''T''HH:mm:ss''Z''','TimeZone','UTC');
load("..\timetables_mat\tt_BK_20180109.mat")
master   = tt_BK_20180109;

elseif strcmp(site,'BK20180726')
DirIn  = 'K:\FLA\BK_20180726';
DirOut = '..\betaout\GTFdet_BK_20180726';
WT = readtable('K:\FLA\FLA_temperature\BK_gov-nps-ever-bkyf1_a830_1301_7798.csv');
WT(isnan(WT.wtemp_C),:) = [];
WT.UTC = datetime(WT.UTC,'InputFormat','yyyy-MM-dd''T''HH:mm:ss''Z''','TimeZone','UTC');
load("..\timetables_mat\tt_BK_20180726.mat")
master   = tt_BK_20180726;

elseif strcmp(site,'JB20180109')
DirIn  = 'K:\FLA\JB_20180109';
DirOut = '..\betaout\GTFdet_JB_20180109';
WT = readtable('K:\FLA\FLA_temperature\JB_gov-nps-ever-jbyf1_a393_7ce3_e678.csv');
WT(isnan(WT.wtemp_C),:) = [];
WT.UTC = datetime(WT.UTC,'InputFormat','yyyy-MM-dd''T''HH:mm:ss''Z''','TimeZone','UTC');
load("..\timetables_mat\tt_JB_20180109.mat")
master   = tt_JB_20180109;

elseif strcmp(site,'JB20151130')
DirIn  = 'K:\FLA\JB_20151130';
DirOut = '..\betaout\GTFdet_JB_20151130';
WT = readtable('K:\FLA\FLA_temperature\JB_gov-nps-ever-jbyf1_a393_7ce3_e678.csv');
WT(isnan(WT.wtemp_C),:) = [];
WT.UTC = datetime(WT.UTC,'InputFormat','yyyy-MM-dd''T''HH:mm:ss''Z''','TimeZone','UTC');
load("..\timetables_mat\tt_JB_20151130.mat")
master   = tt_JB_20151130;

else
DirIn  = 'K:\FLA\LM_20180109';
DirOut = '..\betaout\GTFdet_LM_20180109';
WT = readtable('K:\FLA\FLA_temperature\LM_gov-nps-ever-lmdf1_a830_1301_7798.csv');
WT(isnan(WT.wtemp_C),:) = [];
WT.UTC = datetime(WT.UTC,'InputFormat','yyyy-MM-dd''T''HH:mm:ss''Z''','TimeZone','UTC');
load("..\timetables_mat\tt_LM_20180109.mat")
master   = tt_LM_20180109;
end


% Constants
NSEC    = 120;        % segment length [s]
FsOut   = 24000;      % target sample rate
nfftPow = 13;         % for your spectrogram bin grid
fcenter = 0:FsOut/(2^nfftPow):FsOut;
df      = fcenter(2) - fcenter(1);   % bin width (Hz) — constant across files
col     = @(x) x(:);                 % force column vector
predf0_uncert = 50;


%% ===================== Config =====================
SAVE_IMAGES = true;                 % << turn JPG writing on/off
KEEP_IMAGES = 'all';                % 'all'|'bwhistle'|'other'|'none'

%% ===================== Per-file loop =====================
for m = 1:height(master)
  try
    tic
    master_time = master.t(m); master_time.TimeZone = 'UTC';
    fsraw       = master.fs(m);
    fname       = char(master.fname(m));
    site_file   = strcat(site,'_',sprintf('%03.0f',m));

    % Segment start times (every 2 min across file duration)
    fdurhrs  = (master.nsamp(m)/fsraw)/3600;
    ST       = (master_time:minutes(2):master_time+hours(fdurhrs)-minutes(2))';
    nSeg     = numel(ST);

    fprintf('Processing %s (%d segments)\n', site_file, nSeg);
    if nSeg == 0
        fprintf('  Skipping %s — too short for a full segment.\n', site_file);
        continue
    end

    % --- Vectorized offsets and priors
    offsetSamples = round(fsraw * seconds(ST - master_time));
    segSamples    = round(NSEC * fsraw);

    cwtemp = col(interp1(WT.UTC, WT.wtemp_C, ST, 'linear', 'extrap'));
    predf0 = col(-27.25 + 12.32*cwtemp);
    predf0 = max(130+75, min(470-75, predf0));
    binIdx = max(1, min(numel(fcenter), round(predf0./df) + 1));
    predf0 = col(fcenter(binIdx));
    frange = [max(predf0 - 75, 130), min(predf0 + 75, 470)];   % N×2

    fullIn  = fullfile(DirIn, fname);
    if ~exist(fullIn, 'file')
        warning('wrapper:missingFile', 'Audio file not found, skipping: %s', fullIn);
        continue
    end
    if ~exist(DirOut,'dir'), mkdir(DirOut); end

    filenameIDs = repmat({fname}, nSeg, 1);
    segIDs      = arrayfun(@(i) sprintf('%03d', i), 1:nSeg, 'uni', 0).';

    % ---------- Unified per-file processing: wrapper aggregates ----------
    Btabs  = cell(nSeg,1);
    Otabs  = cell(nSeg,1);
    Dtabs  = cell(nSeg,1);
    Bcount = zeros(nSeg,1);
    Ocount = zeros(nSeg,1);

    usePar = ~isempty(gcp('nocreate'));
    if usePar
        C = parallel.pool.Constant(classifier);
        parfor i = 1:nSeg
            i1 = max(1, offsetSamples(i)+1);
            i2 = min(i1 + segSamples - 1, master.nsamp(m));
            y  = audioread(fullIn, [i1 i2]);
            if isempty(y) || any(isnan(y(:))), continue; end
            if size(y,2) > 1, y = y(:,1); end
            if fsraw ~= FsOut, y = resample(y, FsOut, fsraw); end

            [Bc, Oc, Btab, Otab, Dtab] = gtfDeepCatalogger_ver1( ...
                y, site, segIDs{i}, filenameIDs{i}, DirOut, C.Value, ...
                frange(i,:), predf0(i), predf0_uncert, ST(i), NSEC,...
                'SaveImages', SAVE_IMAGES, 'KeepImages', KEEP_IMAGES, ...
                'UseParInternal', false);

            Btabs{i}=Btab; Otabs{i}=Otab; Dtabs{i}=Dtab;
            Bcount(i)=Bc;  Ocount(i)=Oc;
        end
    else
        for i = 1:nSeg
            i1 = max(1, offsetSamples(i)+1);
            i2 = min(i1 + segSamples - 1, master.nsamp(m));
            y  = audioread(fullIn, [i1 i2]);
            if isempty(y) || any(isnan(y(:))), continue; end
            if size(y,2) > 1, y = y(:,1); end
            if fsraw ~= FsOut, y = resample(y, FsOut, fsraw); end

            [Bc, Oc, Btab, Otab, Dtab] = gtfDeepCatalogger_ver1( ...
                y, site, segIDs{i}, filenameIDs{i}, DirOut, classifier, ...
                frange(i,:), predf0(i), predf0_uncert, ST(i), NSEC,...
                'SaveImages', SAVE_IMAGES, 'KeepImages', KEEP_IMAGES, ...
                'UseParInternal', false);

            Btabs{i}=Btab; Otabs{i}=Otab; Dtabs{i}=Dtab;
            Bcount(i)=Bc;  Ocount(i)=Oc;
        end
    end

    % ---- Combine once per file
    B_all = vertcat(Btabs{cellfun(@istable, Btabs)});
    O_all = vertcat(Otabs{cellfun(@istable, Otabs)});
    D_all = vertcat(Dtabs{cellfun(@istable, Dtabs)});
    if ~isempty(B_all), B_all.abs_time.TimeZone = 'UTC'; end
    if ~isempty(O_all), O_all.abs_time.TimeZone = 'UTC'; end
    if ~isempty(D_all), D_all.abs_time.TimeZone = 'UTC'; end

    % ---- Save once per file
    [~, fbase] = fileparts(fname);
    S = struct();
    S.([fbase '_bwTable'])  = B_all;
    S.([fbase '_oTable'])   = O_all;
    S.([fbase '_DetTable']) = D_all;
    save(fullfile(DirOut, [fbase '_tables.mat']), '-struct', 'S', '-nocompression');

    % ---- SummaryTab from per-segment counts
    Tsum = table(Bcount(:), Ocount(:), repmat(string(site), nSeg,1), ...
                 repmat(m, nSeg,1), repmat(string(fname), nSeg,1), ...
                 (1:nSeg)', ST, ...
                 'VariableNames', {'n_bwhistle','n_other','site','fnumber','file','segnumber','time'});
    varname = matlab.lang.makeValidName(string(fbase) + "_SummaryTab");
    S2 = struct(); S2.(varname) = Tsum;
    save(fullfile(DirOut,[fbase '_SummaryTab.mat']), '-struct', 'S2', '-nocompression');
    fprintf('Done %s in %.1f s\n', site_file, toc);
  catch ME
    fn = 'unknown';
    if exist('fname','var'), fn = fname; end
    fprintf('ERROR file %d (%s): %s\n', m, fn, ME.message);
    if ~isempty(ME.stack)
        fprintf('  at %s  line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
  end
end
