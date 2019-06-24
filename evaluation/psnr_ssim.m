% Refer from: https://github.com/limbee/NTIRE2017/blob/master/demo/evaluation.m
close all; clc;
inputDir = '../data/test/';
outputDir = '../output/';

modelException = {};
setException = {};

scale = 4;
shave = 4;
psnrOnly = false;

disp(repmat('-', 1, 80))
disp([repmat('-', 1, 25), 'SRFeat PSNR & SSIM evaluation', repmat('-', 1, 26)])
disp(repmat('-', 1, 80))
disp(' ')
disp([sprintf('%-25s', 'Model Name'), ' | ', ...
    sprintf('%-10s', 'Set Name'), ' | ', ...
    sprintf('%-5s', 'Scale'), ...
    ' | PSNR / SSIM'])
disp(repmat('-', 1, 80))


totalDir = dir(fullfile(outputDir));
for iModel = 1:length(totalDir)
    modelName = totalDir(iModel).name;
    
    if (modelName(1) == '.') || (any(strcmp(modelException, modelName)) == true)
        continue;
    end
    
    modelFull = fullfile(outputDir, modelName);
    modelDir = dir(modelFull);
    
    for iSet = 1:length(modelDir)
        setName = modelDir(iSet).name;
        if (setName(1) == '.') || (any(strcmp(setException, setName)) == true)
            continue;
        end
        
        setFull = fullfile(modelFull, setName);
        setDir = dir(setFull);
        
        meanPSNR = 0;
        meanSSIM = 0;
        numImages = 0;
        
        for im = 1:length(setDir)
            imageName = setDir(im).name;
            inputName = fullfile(setFull, imageName);
            targetName = fullfile(inputDir, setName, 'HR', imageName);
            if (imageName(1) ~= '.') && (strcmp(imageName, 'Thumbs.db') == 0) && (exist(targetName, 'file') == 2)
                inputImg = imread(inputName);
                targetImg = imread(targetName);
                targetDim = length(size(targetImg));
                if targetDim == 2
                    targetImg = cat(3, targetImg, targetImg, targetImg);
                end
                
                if sum(strcmp(setName, {'Set5', 'Set14', 'BSDS100'})) == 1
%                     if targetDim == 2
%                         targetImg = targetImg(:,:,1);
%                         inputImg = inputImg(:,:,1);
%                     else
                        targetImg = rgb2ycbcr(targetImg);
                        targetImg = targetImg(:,:,1);
                        inputImg = rgb2ycbcr(inputImg);
                        inputImg = inputImg(:,:,1);
%                     end
                end
                [h, w, ~] = size(inputImg);
                inputImg = inputImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
                targetImg = targetImg((1 + shave):(h - shave), (1 + shave):(w - shave), :);
                meanPSNR = meanPSNR + psnr(inputImg, targetImg);
                if psnrOnly == false
                    meanSSIM = meanSSIM + ssim(inputImg, targetImg);
                end
                numImages = numImages + 1;
            end
        end
        if (numImages > 0)
            meanPSNR = meanPSNR / numImages;
            meanSSIM = meanSSIM / numImages;
            
            modelNameF = sprintf('%-25s', modelName);
            setNameF = sprintf('%-10s', setName);
            scaleF = sprintf('%-5d', scale);
            isModelPrint = true;
            isSetPrint = true;
            disp([modelNameF, ' | ', ...
            setNameF, ' | ', ...
            scaleF, ...
            ' | PSNR: ', num2str(meanPSNR, '%.2fdB')])
            if psnrOnly == false
                disp([repmat(' ', 1, 25), ' | ', ...
                repmat(' ', 1, 10), ' | ', ...
                repmat(' ', 1, 5), ...
                ' | SSIM: ', num2str(meanSSIM, '%.4f')])
            end
        end
    end
    disp(repmat('-', 1, 80))
end