truth = readmatrix('GroundTruth-numbered-only.csv');
tmpltTruth = readmatrix('output/output.csv');

score = 0;

% loop through data in truth for template matching
for i = 1 : size(tmpltTruth, 1)
    % loop through ground truth
    for j = 1 : size(truth, 1)
        % if match, add to IOU score
        if tmpltTruth(i, 1) == truth(j, 1)
            % box from ground truth
            boxTruth = [truth(j, 4) truth(j, 2) truth(j, 5)-truth(j, 4) truth(j, 3)-truth(j, 2)];
            % box from template matching truth
            boxTmplt = [tmpltTruth(i, 4) tmpltTruth(i, 2) tmpltTruth(i, 5)-tmpltTruth(i, 4) tmpltTruth(i ,3)-tmpltTruth(i, 2)];
            
            % save overlap
            score = score + bboxOverlapRatio(boxTruth, boxTmplt);
            
            break
        end
    end
end

% display average (a score of 1 is perfect)
disp(score/50);