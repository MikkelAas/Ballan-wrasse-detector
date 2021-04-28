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

    % binaryImage = imfill(binaryImage, 'holes');

    % Remove disturbances from the binary image
    rectElem = strel('rectangle',[2,2]);
    Ibwopen = imopen(binaryImage, rectElem);

    hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea', 10000);
    [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);

    % If no bbox is found, try another mask
    if isempty(bboxOut)
         % Find a binary image using thresholding
        [binaryImage, imageMasked] = createMask2(image);

        % Remove disturbances from the binary image
        Ibwopen = imopen(binaryImage, rectElem);

        hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea', 2000);
        [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);
    end

    % If no bbox is found, try another mask
    if isempty(bboxOut)
        % Find a binary image using thresholding
        [binaryImage, imageMasked] = createMask3(image);

        % Remove disturbances from the binary image
        Ibwopen = imopen(binaryImage, rectElem);

        hBlobAnalysis = vision.BlobAnalysis('MinimumBlobArea', 2000);
        [objArea, objCentroid, bboxOut] = step(hBlobAnalysis, Ibwopen);
    end

    % If there still is not a bbox found, give up
    if isempty(bboxOut)
        disp("No bounding box found")
        overlapRatio = 0;
    else
        disp("Bounding box found")
        overlapRatioArray = bboxOverlapRatio(bboxOut, bboxFish(k, :));
        overlapRatio = 0;
        for i=1:length(overlapRatio)
            if(overlapRatioArray(i) > overlapRatio)
                overlapRatio = overlapRatioArray(i);
            end
        end
    end
    IoUScore(k) = overlapRatio;
    overlapTotal = overlapTotal + overlapRatio;
    bboxCounter = bboxCounter + 1;

    
    % Add rectangle to the image
    %Ishape = insertShape(image, 'Rectangle', bboxOut,'Linewidth',4, 'Color', 'red');


    %imwrite(Ishape, "/home/ukhu/Documents/fisk/output/"+baseFileName)

end
toc
IoUScore(2, :) = [0.05386178861788618, 0.305043964403871, 0.031221719457013575, 0.19455858787220745, 0.25469990437857193, 0.7571122667482411, 0.7267085811384877, 0.2325383629278942, 0.5380407788402365, 0.18506807202459377, 0.04330478785519658, 0.7839125635986756, 0.07108860759493671, 0.08869242467201072, 0.23339019600213798, 0.6291368701147514, 0.2710677841468606, 0.2521775496077526, 0.15087085079040027, 0.19292482557458604, 0.0, 0.24464434797128068, 0.19643748113072632, 0.32790167717083496, 0.3720391108452205, 0.13327514974674942, 0.09309149258474425, 0.4533898305084746, 0.23439256825351776, 0.14632460938046413, 0.6810067634992792, 0.10672042345840857, 0.14790101447429857, 0.041016876735740226, 0.1447372833851564, 0.14741835431490605, 0.3538256124463021, 0.39023046552199064, 0.1796276747264532, 0.2785295853957586, 0.7438994979281314, 0.1680327190237279, 0.05293263682950132, 0.1836384771868643, 0.2511574074074074, 0.6209850107066381, 0.24679836159070878, 0.18906154895389513, 0.7174672489082969, 0.22900459688826025];
IoUScore = IoUScore'
boxplot(IoUScore)
overlapAverage = (overlapTotal / bboxCounter)
