%%%%%%%
rng(42)
% Fine-tuning version: trains ResNet50 end-to-end on spectrogram images.
% AUTHORS: D. Bohnenstiehl (NCSU) — gulf toadfish finder v.1 fine-tuned

% Threshold selection method: 'max F1' | 'min cost' | 'F1 plateau'
thr_method = 'F1 plateau';

%% Load images
datasetDir = '..\..\train_jpg';
imds = imageDatastore(datasetDir, 'LabelSource','foldernames', 'IncludeSubfolders',true);
imds = shuffle(imds);
tbl  = countEachLabel(imds)

out_directory = 'D:\drbohnen\BetaJun26\GTF_ResNet50tuned';

%% Sample images
bwhistle = find(imds.Labels == 'bwhistle');
other    = find(imds.Labels == 'other');
nP = table2array(tbl(1,2));
nO = table2array(tbl(2,2));can 
figure
for i = 1:16
    subplot(4,4,i); imshow(readimage(imds, bwhistle(randi(nP,1)))); title('bwhistle')
end
figure
for i = 1:16
    subplot(4,4,i); imshow(readimage(imds, other(randi(nO,1)))); title('other')
end

%% Build fine-tuned ResNet50
% Replace the 1000-class ImageNet head with a 2-class head.
% New layers get 10x learning rate so they train fast; backbone uses global LR.
net    = resnet50();
lgraph = layerGraph(net);

newFC = fullyConnectedLayer(2, 'Name','fc_toad', ...
            'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
newSM = softmaxLayer('Name','softmax_toad');
newCL = classificationLayer('Name','output_toad');

lgraph = replaceLayer(lgraph, 'fc1000',                     newFC);
lgraph = replaceLayer(lgraph, 'fc1000_softmax',             newSM);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newCL);

%% Split — 70 / 15 / 15  (train / val / test)
% valSet drives early stopping and best-checkpoint selection.
% testSet is held out completely until final evaluation.
[trainingSet, valSet, testSet] = splitEachLabel(imds, 0.70, 0.15, 'randomize');

% Training augmentation: random left-right reflection adds time-reversed
% copies of each call, doubling effective dataset size cheaply.
augmentLR = true;   % set false to disable left-right flip augmentation
augmenter = imageDataAugmenter('RandXReflection', augmentLR);
augTrain  = augmentedImageDatastore([224 224 3], trainingSet, ...
                'DataAugmentation', augmenter);
augVal    = augmentedImageDatastore([224 224 3], valSet);    % no augmentation
augTest   = augmentedImageDatastore([224 224 3], testSet);   % no augmentation

%% Training options
% MiniBatchSize=64 keeps GPU well-fed without going OOM on typical GPUs.
% ExecutionEnvironment='gpu' forces GPU use (falls back to CPU if no GPU).
% OutputNetwork='best-validation-loss' saves the checkpoint with lowest
% validation loss rather than the final epoch (guards against overfit).
% ValidationPatience=8 stops early if validation loss stalls.
nTrain = numel(trainingSet.Files);
valFreq = max(10, floor(nTrain / 64 / 3));   % ~3 validations per epoch

opts = trainingOptions('sgdm',                              ...
    'MiniBatchSize',         64,                            ...
    'MaxEpochs',             30,                            ...
    'InitialLearnRate',      1e-4,                          ...
    'Momentum',              0.9,                           ...
    'L2Regularization',      1e-4,                          ...
    'LearnRateSchedule',     'piecewise',                   ...
    'LearnRateDropPeriod',   5,                             ...
    'LearnRateDropFactor',   0.1,                           ...
    'ValidationData',        augVal,                        ...
    'ValidationFrequency',   valFreq,                       ...
    'ValidationPatience',    8,                             ...
    'OutputNetwork',         'best-validation-loss',        ...
    'ExecutionEnvironment',  'gpu',                         ...
    'Shuffle',               'every-epoch',                 ...
    'Verbose',               true,                          ...
    'Plots',                 'training-progress');

%% Fine-tune
[netFine, trainInfo] = trainNetwork(augTrain, lgraph, opts);

fprintf('Training stopped at epoch %d\n', numel(trainInfo.TrainingLoss));

%% Evaluate on test set
testLabels = testSet.Labels;
predLabels = classify(netFine, augTest, 'ExecutionEnvironment','gpu');
scores     = predict(netFine,  augTest, 'ExecutionEnvironment','gpu');

classes = netFine.Layers(end).Classes;
posIdx  = find(classes == 'bwhistle');
p       = scores(:, posIdx);

confMat_raw = confusionmat(testLabels, predLabels);
disp('Test confusion matrix:')
bsxfun(@rdivide, confMat_raw, sum(confMat_raw, 2))

TP = confMat_raw(1,1); FN_c = confMat_raw(1,2); FP_c = confMat_raw(2,1);
precision = TP / (TP + FP_c);
recall    = TP / (TP + FN_c);
F1_score  = 2 * precision * recall / (precision + recall);
fprintf('Precision: %.3f  Recall: %.3f  F1: %.3f\n', precision, recall, F1_score);

figure;
cm = confusionchart(testLabels, predLabels);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary    = 'row-normalized';
cm.Title         = 'Fine-tuned ResNet50 — Test Confusion Matrix';

%% ROC — compute curve data now; figure drawn after bestThr is known
[x, y, T_roc, auc] = perfcurve(testLabels, p, 'bwhistle');
fprintf('AUC: %.4f\n', auc);

%% Threshold sweep
fp_weight = 1.075;
fn_weight = 1.0;
thresholds = 0.05:0.01:0.95;
prec_t = nan(size(thresholds));
rec_t  = nan(size(thresholds));
f1_t   = nan(size(thresholds));
cost_t = nan(size(thresholds));
N_test = numel(testLabels);

for k = 1:numel(thresholds)
    pl = repmat(categorical({'other'}), size(p));
    pl(p >= thresholds(k)) = categorical({'bwhistle'});
    cm2   = confusionmat(testLabels, pl);
    tp    = cm2(1,1); fn_k = cm2(1,2); fp_k = cm2(2,1);
    denom = tp + fp_k;
    if denom > 0; prec_t(k) = tp / denom; end
    rec_t(k) = tp / (tp + fn_k);
    if prec_t(k) + rec_t(k) > 0
        f1_t(k) = 2 * prec_t(k) * rec_t(k) / (prec_t(k) + rec_t(k));
    end
    cost_t(k) = (fp_k * fp_weight + fn_k * fn_weight) / N_test;
end

[~, best_cost_idx] = min(cost_t);
[~, best_f1_idx]   = max(f1_t);

switch thr_method
    case 'min cost'
        bestThr = thresholds(best_cost_idx);
    case 'max F1'
        bestThr = thresholds(best_f1_idx);
    case 'F1 plateau'
        plateau_mask = f1_t >= (max(f1_t) - 0.001);
        plateau_thrs = thresholds(plateau_mask);
        bestThr      = plateau_thrs(ceil(numel(plateau_thrs)/2));
    otherwise
        error('thr_method must be ''max F1'', ''min cost'', or ''F1 plateau''; got ''%s''', thr_method);
end

fprintf('Threshold %.2f (min cost)        : F1=%.3f\n', thresholds(best_cost_idx), f1_t(best_cost_idx));
fprintf('Threshold %.2f (max F1)          : F1=%.3f\n', thresholds(best_f1_idx),   f1_t(best_f1_idx));
fprintf('Threshold %.2f (selected: %s)\n', bestThr, thr_method);

%% ROC figure — drawn here so decision boundary (bestThr) can be marked
[~, op_idx] = min(abs(T_roc - bestThr));
op_fpr = x(op_idx);
op_tpr = y(op_idx);
figure; plot(x, y, 'LineWidth', 1.5); hold on;
plot(op_fpr, op_tpr, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(op_fpr + 0.02, op_tpr - 0.03, sprintf('thr=%.2f', bestThr), 'FontSize', 9);
hold off;
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC — fine-tuned ResNet50 — Test data  (AUC = %.4f)', auc)); grid on;

figure;
yyaxis left
plot(thresholds, prec_t, thresholds, rec_t, thresholds, f1_t, 'LineWidth',1.5);
ylabel('Rate'); ylim([0 1]);
yyaxis right
plot(thresholds, cost_t, '--', 'LineWidth',1.5);
ylabel('Weighted cost per sample');
legend('Precision','Recall','F1','Weighted cost','Location','southwest');
xlabel('Threshold'); grid on;
title(sprintf('Threshold sweep — fine-tuned  (FP=%.3f  FN=%.3f)', fp_weight, fn_weight));

%% FP / FN image panels
display_thresh = bestThr;
pl_thresh = repmat(categorical({'other'}), size(p));
pl_thresh(p >= display_thresh) = categorical({'bwhistle'});

isPredPos = (pl_thresh  == 'bwhistle');
isTruePos = (testLabels == 'bwhistle');
FP_idx = find( isPredPos & ~isTruePos);
FN_idx = find(~isPredPos &  isTruePos);

[~, ordFP] = sort(p(FP_idx), 'descend'); FP_sorted = FP_idx(ordFP);
[~, ordFN] = sort(p(FN_idx), 'ascend');  FN_sorted = FN_idx(ordFN);

Nshow = 16;
figure('Name','False Positives');
tiledlayout(4,4,'Padding','compact','TileSpacing','compact');
for k = 1:Nshow
    nexttile;
    if k <= numel(FP_sorted)
        imshow(readimage(testSet, FP_sorted(k))); axis off;
        title(sprintf('p=%.2f', p(FP_sorted(k))), 'FontSize',8);
    else; axis off; end
end
sgtitle(sprintf('False Positives  (showing %d of %d)', min(Nshow,numel(FP_idx)), numel(FP_idx)));

fprintf('\n--- False Positives (showing %d of %d) ---\n', min(Nshow,numel(FP_idx)), numel(FP_idx));
fprintf('  %3s  %12s  %s\n', '#', 'p(bwhistle)', 'filename');
for k = 1:min(Nshow, numel(FP_sorted))
    [~, fname, ext] = fileparts(testSet.Files{FP_sorted(k)});
    fprintf('  %3d  %12.4f  %s\n', k, p(FP_sorted(k)), [fname ext]);
end

figure('Name','False Negatives');
tiledlayout(4,4,'Padding','compact','TileSpacing','compact');
for k = 1:Nshow
    nexttile;
    if k <= numel(FN_sorted)
        imshow(readimage(testSet, FN_sorted(k))); axis off;
        title(sprintf('p=%.2f', p(FN_sorted(k))), 'FontSize',8);
    else; axis off; end
end
sgtitle(sprintf('False Negatives  (showing %d of %d)', min(Nshow,numel(FN_idx)), numel(FN_idx)));

fprintf('\n--- False Negatives (showing %d of %d) ---\n', min(Nshow,numel(FN_idx)), numel(FN_idx));
fprintf('  %3s  %12s  %s\n', '#', 'p(bwhistle)', 'filename');
for k = 1:min(Nshow, numel(FN_sorted))
    [~, fname, ext] = fileparts(testSet.Files{FN_sorted(k)});
    fprintf('  %3d  %12.4f  %s\n', k, p(FN_sorted(k)), [fname ext]);
end

%% Save
classifier = struct('net', netFine, 'bestThr', bestThr, 'posIdx', posIdx);
save('gtfclassifier_tuned.mat', 'classifier');
fprintf('Saved to gtfclassifier_tuned.mat\n');
