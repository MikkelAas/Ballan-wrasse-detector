clear

myFolder = '/home/ukhu/Documents/fisk/testing-numbered';
filePattern = fullfile(myFolder, '*.jpg');
theFiles = dir(filePattern);

truth = readmatrix('Documents/fisk/GroundTruth-numbered-only.csv');

IoUScores = [];
bboxFish = zeros(length(theFiles), 4);
overlapTotal = 0;
bboxCounter = 0;

tic
for i = 1:height(truth)
    % Gets the bounding box from the ground truth
    bboxFish(truth(i,1), :) = [truth(i, 4), truth(i, 2), truth(i,5)-truth(i,4), truth(i,3)-truth(i,2)];
end


for k = 1:length(theFiles)
    
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    image = imread(fullFileName);

    % Find a binary image using thresholding
    [binaryImage, imageMasked] = createMask(image);



    % Remove disturbances from the binary image
    diskElem = strel('disk',2);
    Ibwopen = imopen(binaryImage, diskElem);
    Ibwopen = imfill(Ibwopen, 'holes');

    hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea',10000);
    [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);

    % If no bbox is found, try another mask
    if isempty(bboxOut)
         % Find a binary image using thresholding
        [binaryImage, imageMasked] = createMask2(image);


        % Remove disturbances from the binary image
        Ibwopen = imopen(binaryImage, diskElem);
        Ibwopen = imfill(Ibwopen, 'holes');

        hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea',10000);
        [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);
    end
    % If no bbox is found, try another mask
    if isempty(bboxOut)
        % Find a binary image using thresholding
        [binaryImage, imageMasked] = createMask3(image);


        % Remove disturbances from the binary image
        Ibwopen = imopen(binaryImage, diskElem);
        Ibwopen = imfill(Ibwopen, 'holes');

        hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea',10000);
        [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);
    end

    % If there still is not a bbox found, give up
    if isempty(bboxOut)
        disp("No bounding box found")
        overlapRatio = 0;
    else
        disp("Bounding box found")
        overlapRatio = bboxOverlapRatio(bboxOut, bboxFish(k, :));
        overlapRatio = overlapRatio(1, :);
        IoUScore(k) = overlapRatio;
    end
    overlapTotal = overlapTotal + overlapRatio;
    bboxCounter = bboxCounter + 1;

    
    % Add rectangle to the image
    %Ishape = insertShape(image, 'Rectangle', bboxOut,'Linewidth',4, 'Color', 'red');
    %Ishape = insertShape(Ishape, 'FilledRectangle', bboxFish,'Linewidth',4, 'Color', 'green');


    %imwrite(Ishape, "/home/ukhu/Documents/fisk/output/"+baseFileName)

end
toc
boxplot(IoUScore)
overlapAverage = (overlapTotal / bboxCounter)
