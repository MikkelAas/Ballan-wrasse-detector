template = rgb2gray(imread('fish_croped.png'));

% the templates scale compared to original image
% (this can really be anything, but for now this is something)
templateScaleHor = 0.323958333;
templateScaleVer = 0.205555556;

% find the files
myFolder = '/home/jakob/NTNU/4. Semester/IDATG2206 - Computer Vision/Project/split_dataset/testing-numbered';
filePattern = fullfile(myFolder, '*.jpg');
theFiles = dir(filePattern);

% file to save results in (might need to be created first? idk)
fileID = fopen('output/output.csv','w');

%do the stuff
tic
for k = 1 : length(theFiles)
    % load image
    [folder, baseFileName, extension] = fileparts(fullfile(theFiles(k).folder, theFiles(k).name));
    image = rgb2gray(imread(fullfile(folder, append(baseFileName,extension))));
    
    try
        % try to find match
        rectanglePosAndSize = tempmatchcrosscorr(image, template);
        
        % save image with rectangle drawn around fish :)
        figure('visible','off');
        imshow(image);
        
        rectangle('position', rectanglePosAndSize, 'edgecolor', 'g', 'linewidth',2);
        saveas(gcf,"./output/" + baseFileName + extension);
        disp("oh yeah: " + baseFileName);
        
        % save detection area in the same format as ground truth
        fprintf(fileID, ...
            '%s,%d,%d,%d,%d\n', ...
            baseFileName, ...
            rectanglePosAndSize(2), ... % y1
            rectanglePosAndSize(2)+rectanglePosAndSize(4), ... %y2
            rectanglePosAndSize(1), ... % x1
            rectanglePosAndSize(1)+rectanglePosAndSize(3) ... % x2
        );
        err = false;
    catch error
        disp("uh oh: "  + baseFileName);
    end
end
toc

fclose(fileID);