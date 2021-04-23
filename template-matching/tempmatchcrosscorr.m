function rectanglePosAndSize = tempmatchcrosscorr(image, template)
% save size of template for later use
dimensionsTemplate = size(template);

% calculate "correlation score" between the images
correlation = normxcorr2(template, image);

% find the cooridinates for the highest score 
% (this will only find the first one even if there are more exact matches)
[~, index] = max(abs(correlation(:)));
[y, x] = ind2sub(size(correlation),index(1));

% (wack)
% it seems that the starting x-coordinate for matched area is 
% negative if it didn't find a good match.
% (this can wrongly mark a good match as bad if the match is close to left edge)
%if (x-dimensionsTemplate(2)) < 0
%    disp("No good match found");
%    return
%end

% according to documentation for normxcorr2, some padding that needs to
% be compensated for when drawing rectangle
rectanglePosAndSize = [x-dimensionsTemplate(2) y-dimensionsTemplate(1) dimensionsTemplate(2), dimensionsTemplate(1)];

