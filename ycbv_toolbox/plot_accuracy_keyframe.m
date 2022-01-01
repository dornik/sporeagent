function plot_accuracy_keyframe

color = {'r', 'y', 'g', 'b', 'm'};
leng = {'SporeAgent'};
aps = zeros(1, 1);
lengs = cell(1, 1);
close all;

% load results
object = load('results_sporeagent-ilrl.mat');
distances_sys = object.distances_sys;
distances_non = object.distances_non;
distances_ad = object.distances_ad;
rotations = object.errors_rotation;
translations = object.errors_translation;
cls_ids = object.results_cls_id;

index_plot = [1];

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
classes = C{1};
classes{end+1} = 'All 21 objects';
fclose(fid);

% hf = figure('units','normalized','outerposition',[0 0 1 1]);
% font_size = 24;
max_distance = 0.1;

ADD = zeros(22, 1);
AD = zeros(22, 1);
ADI = zeros(22, 1);

% for each class
for k = 1:numel(classes)
    index = find(cls_ids == k);
    if isempty(index)
        index = 1:size(distances_sys,1);
    end

    % distance symmetry
    subplot(2, 2, 1);
    for i = index_plot
        D = distances_sys(index, i);
        D(D > max_distance) = inf;
        d = sort(D);
        n = numel(d);
        accuracy = cumsum(ones(1, n)) / n;        
        plot(d, accuracy, color{i}, 'LineWidth', 4);
        aps(i) = VOCap(d, accuracy);
        ADI(k) = aps(i);
        lengs{i} = sprintf('%s (%.2f)', leng{i}, aps(i) * 100);
        hold on;
    end
    hold off;
    %h = legend('network', 'refine tranlation only', 'icp', 'stereo translation only', 'stereo full', '3d coordinate');
    %set(h, 'FontSize', 16);
    h = legend(lengs(index_plot), 'Location', 'southeast');
    set(h, 'FontSize', font_size);
    h = xlabel('Average distance threshold in meter (symmetry)');
    set(h, 'FontSize', font_size);
    h = ylabel('accuracy');
    set(h, 'FontSize', font_size);
    h = title(classes{k}, 'Interpreter', 'none');
    set(h, 'FontSize', font_size);
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', font_size)

    % distance non-symmetry
    subplot(2, 2, 2);
    for i = index_plot
        D = distances_non(index, i);
        D(D > max_distance) = inf;
        d = sort(D);
        n = numel(d);
        accuracy = cumsum(ones(1, n)) / n;
        plot(d, accuracy, color{i}, 'LineWidth', 4);
        aps(i) = VOCap(d, accuracy);
        ADD(k) = aps(i);
        lengs{i} = sprintf('%s (%.2f)', leng{i}, aps(i) * 100);        
        hold on;
    end

    % distance (non-)symmetry
    for i = index_plot
        D = distances_ad(index, i);
        D(D > max_distance) = inf;
        d = sort(D);
        n = numel(d);
        accuracy = cumsum(ones(1, n)) / n;
        aps(i) = VOCap(d, accuracy);
        AD(k) = aps(i);
        lengs{i} = sprintf('%s (%.2f)', leng{i}, aps(i) * 100);
    end
     
    filename = sprintf('plots/%s.png', classes{k});
    hgexport(hf, filename, hgexport('factorystyle'), 'Format', 'png');
end
ADD_per_class = ADD(1:end-1)
AD_per_class = AD(1:end-1)
ADI_per_class = ADI(1:end-1)

ADD_mean_over_classes = mean(ADD(1:end-1))
AD_mean_over_classes = mean(AD(1:end-1))
ADI_mean_over_classes = mean(ADI(1:end-1))


function ap = VOCap(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';

mrec=[0 ; rec ; 0.1];
mpre=[0 ; prec ; prec(end)];
for i = 2:numel(mpre)
    mpre(i) = max(mpre(i), mpre(i-1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) * 10;
