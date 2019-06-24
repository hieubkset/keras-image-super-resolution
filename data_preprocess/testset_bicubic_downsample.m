clear; close all; clc;

scale = 4;

folder = '../data/test/BSDS100/';
og_folder = strcat(folder, 'OG/');
hr_folder = strcat(folder, 'HR/');
lr_folder = strcat(folder, 'LR/');

if ~exist(og_folder, 'dir')
    fprintf('not found source folder!\n')
    return
end

if ~exist(lr_folder, 'dir')
       mkdir(lr_folder)
end

if ~exist(hr_folder, 'dir')
       mkdir(hr_folder)
end

filepaths = dir(fullfile(og_folder, '*.png'));

for i = 1 : length(filepaths)
    image = imread(fullfile(og_folder, filepaths(i).name));
    
    [H, W, C] = size(image);
    h_mod = mod(H, scale);
    w_mod = mod(W, scale);
    
    hr_image = imcrop(image, [1, 1, W-w_mod, H-h_mod]);
    lr_image = imresize(hr_image, 1/scale, 'bicubic');
    
    imwrite(hr_image, fullfile(hr_folder, [filepaths(i).name(1:end-4) '.png'])); 
    imwrite(lr_image, fullfile(lr_folder, [filepaths(i).name(1:end-4) '.png'])); 
    fprintf(filepaths(i).name); fprintf("\n");
end
