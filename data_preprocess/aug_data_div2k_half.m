clear;close all;
% settings
folder = '../data/train/DIV2K/origin/';
savepath = '../data/train/DIV2K/';
size_input = 74;
%scale 
scale = 4; 
size_label = size_input * scale;
stride = size_input * scale;



% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
padding = 0;
count = 0;
do_write = 0;

totalct = 0;
created_flag = false;
change =0;
old =1;
% generate data
filepaths = dir(fullfile(folder,'*.png'));

rand_index = randperm(length(filepaths));
c=0
for index = 1 : length(filepaths)

        i = rand_index(index);
        filepaths(i).name
        image = imread(fullfile(folder,filepaths(i).name));
        image = imresize(image,1/2,'bicubic');

        [H, W, C] = size(image);
        [add, im_name, type] = fileparts(filepaths(i).name);
        left = floor(( W - size_label)/2)+1 ;
        top = floor((H - size_label)/2) +1;
        image1 = image;
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end
        

        image1 = flip(image,2);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end
        
        image1 = imrotate(image,90);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']
        

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end
        
        image1 = imrotate(image,180);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']
        
        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end
        
        image1 = imrotate(image,270);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end
        
        
        image1 = flip(image,2);
        image1 = imrotate(image1,90);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end        
        
        
        image1 = flip(image,2);
        image1 = imrotate(image1,180);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end                
        
        image1 = flip(image,2);
        image1 = imrotate(image1,270);
        for kk = 1:5
        do_write = do_write +1;
        fprintf('do_write: %d\n', do_write);
        outname = [filepaths(i).name(1:end-5) '.png']

        y_s = randsample(1:size(image1,1)-size_label+1,1);
        x_s = randsample(1:size(image1,2)-size_label+1,1);
        subim_label2 = imcrop(image1, [x_s,y_s,size_label-1,size_label-1]);
        subim_input2 = imresize(subim_label2,1/scale,'bicubic');
        imwrite(subim_label2, fullfile(savepath,'HR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));
        imwrite(subim_input2, fullfile(savepath,'LR',[filepaths(i).name(1:end-5) '_' num2str(do_write) '_2.png']));   
        end                
end