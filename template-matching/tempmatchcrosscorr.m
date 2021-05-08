function rectanglePosAndSize = tempmatchcrosscorr(image, template, threshold)

% save size of template for later use
dimensionsTemplate = size(template);

% calculate "correlation score" between the images

% loop that will shrink the template until it fits in image 
% not pretty but works :)
error = true;
while error
    try
        correlation = normxcorr2(template, image);
        error = false;
    catch
        template = imresize(template, 0.75);
    end
end


% find the cooridinates for the highest score 
% (this will only find the first one even if there are more exact matches)
[value, index] = max(abs(correlation(:)));
[y, x] = ind2sub(size(correlation),index(1));

if value >= threshold
    % according to documentation for normxcorr2, some padding that needs to
    % be compensated for when drawing rectangle
    rectanglePosAndSize = [x-dimensionsTemplate(2) y-dimensionsTemplate(1) dimensionsTemplate(2), dimensionsTemplate(1)];
else
    % it didn't meet the threshold value, set all rectangle values to -1
    rectanglePosAndSize = [-1 -1 -1 -1];
end
