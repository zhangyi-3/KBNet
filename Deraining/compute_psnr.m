function psnr=compute_psnr(img1,img2)
    if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
    end

    if size(img2, 3) == 3
        img2 = rgb2ycbcr(img2);
        img2 = img2(:, :, 1);
    end

    imdff = double(img1) - double(img2);
    imdff = imdff(:);
    rmse = sqrt(mean(imdff.^2));
    psnr = 20*log10(255/rmse);

end