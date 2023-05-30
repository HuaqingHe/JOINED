close all;
clear

basename = '.\PALM\Train\newdata\';
subdir = dir(basename);
% for s = 1 : length(subdir)
% 如果不是目录，则跳过
%  || ~subdir(s).isdir 这个是如果是文件而不是目录就跳过
%     if isequal(subdir(s).name, '.') || isequal(subdir(s).name, '..')
%         continue;
%     end
% end
Files = dir([basename 'mask_DC\*.bmp']);
[fovea_location, ~] = xlsread([basename 'newLocalization_Results.csv']);
number = length(Files);
%% 每个照片都来做一次  把视杯视盘弄到一起，做两个个新label，上面只有视盘和黄斑区域的高斯 Fovea1 OD2 background3
%   并且 需要另外一个label，是做一个视盘和黄斑之间的距离的图像，有些类似
for i = 1 : number
    mask = imread([basename 'mask_DC\' Files(i).name]);
    [h, w] = size(mask);
    
    % 这里为了把边角上的0 变成255 避免造成误导
    mask_fill = imfill(mask, 'holes');
    [L,num]=bwlabel(mask_fill,4);
    mask(L == 0) = 255;
    % 找fovea中心
    %     [X_fovea, Y_fovea] = find(mask == 192);
    %     x_fovea = round((max(X_fovea) + min(X_fovea)) / 2);
    %     y_fovea = round((max(Y_fovea) + min(Y_fovea)) / 2);
    x_fovea = round(fovea_location(i, 3));
    y_fovea = round(fovea_location(i, 2));
    [X_OD, Y_OD] = find(mask == 0);
    x_OD = round((max(X_OD) + min(X_OD)) / 2);
    y_OD = round((max(Y_OD) + min(Y_OD)) / 2);
    %% 开始造高斯label
    sigma_fovea = h/20;  %影响半径,这里最好改成相对路径
    sigma_OD = h/20;
    x = 1:1:h;
    y = 1:1:w;
    [X, Y] = meshgrid(y, x);
    mask_optic = zeros(h, w);
    mask_optic = uint8(mask_optic);             % 直接uint8会导致只出现[255,255]
    Zsum = zeros(size(X));
    Z_fovea = zeros(size(X));
    if x_fovea > 0 && y_fovea > 0
        Z_fovea = Gauss2D(X, Y, sigma_fovea, y_fovea, x_fovea);
        Z_fovea(Z_fovea<0.0001) = 0;
        % Z_fovea = -1 - rescale(Z_fovea, -1, 0);           % 把高斯变到-1-0  用以区别fovea和OD
        Z_fovea = rescale(Z_fovea, 0, 1);                      % 不影响欸
    end

    Z_OD = zeros(size(X));
    if ~isempty(x_OD)
        Z_OD = Gauss2D(X, Y, sigma_OD, y_OD, x_OD);
        Z_OD(Z_OD<0.0001) = 0;
        Z_OD = rescale(Z_OD, 0, 1);
    end
    mask_F = Heatsum(Zsum, Z_fovea);
    mask_D = Heatsum(Zsum, Z_OD);
    mask_F(mask_F < 0.01) = 0;
    mask_D(mask_D < 0.01) = 0;
    mask_FD = Heatsum(mask_F, mask_D);
%     figure;
%     mesh(X,Y,mask_F,'FaceColor','W');
%     xlabel('x');ylabel('y');zlabel('z');
%     figure;
%     mesh(X,Y,mask_FD,'FaceColor','W');
%     xlabel('x');ylabel('y');zlabel('z');
    % 保存成图片再读入，会导致精度下降，比如这里就会有很多个最大值，直接保存mat算了
%     imwrite(mask_FD,[basename '\mask_FD_0.05h\' Files(i).name(1:end-4) '.bmp']);
    filename = [basename '\mask_FD_mat_0.05h\' Files(i).name(1:end-4) '.mat'];
    save(filename,'mask_F','mask_D');
    
    
    %% 做distlabel       https://www.mathworks.com/help/images/ref/bwdist.html
    bw = zeros(h, w);
    flag = 0;
    if x_fovea > 0 && y_fovea > 0   
        bw(x_fovea, y_fovea) = 1;
    else
        disp([num2str(i) 'no fovea'])
        flag = 1;
    end
    
    if ~isempty(x_OD)
        bw(x_OD, y_OD) = 1;
    else
        disp([num2str(i) 'no OD'])
        if flag == 1
            bw = ones(h, w);
        end
    end
%             bw2 = zeros(h, w);
%             bw2(x_OD, y_OD) = 1;
%             [mask_distF, ~] = bwdist(bw, 'euclidean');
%             [mask_distD, ~] = bwdist(bw2, 'euclidean');
%             RGB1 = repmat(rescale(mask_dist1), [1 1 3]);
%             imshow(RGB1), title('Euclidean')
%             hold on, imcontour(mask_dist1)
% 
%             mask_distF = rescale(mask_distF, 0, 1);
%             mask_distD = rescale(mask_distD, 0, 1);
    [mask_distFD, ~] = bwdist(bw, 'euclidean');
    mask_distFD = rescale(mask_distFD, 0, 1); 
    save([basename 'mask_FDdist_mat\' Files(i).name(1:end-4) '.mat'], 'mask_distFD');
%         figure;
%         mesh(X,Y,mask_distFD,'FaceColor','W');
%         hold on, imcontour(X, Y, mask_distFD);
%         xlabel('x');ylabel('y');zlabel('z');
        %         figure;
        %         mesh(X,Y,mask_dist,'FaceColor','W');
        %         figure;
        %         surf(X,Y,mask_dist,'EdgeAlpha',0.7,'FaceAlpha',0.9)

        %         imwrite(mask_dist1,[basename '\mask_dist_mat\' Files(i).name(1:end-4) '.bmp']);
    
end


function Z=Gauss2D(X,Y,sigma,a,b)
% XY是网格坐标,sigma是高斯分布的宽度,ab是中心点坐标
Z=0.5*exp(-((X-a).^2+(Y-b).^2)./sigma.^2);
end


function B=Heatsum(A1,A2)
%两个点之间叠加
B=1-(1-A1).*(1-A2);
end
