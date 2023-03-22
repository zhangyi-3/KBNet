function ssim_mean=compute_ssim(img1,img2)
    if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
    end

    if size(img2, 3) == 3
        img2 = rgb2ycbcr(img2);
        img2 = img2(:, :, 1);
    end
    ssim_mean = SSIM_index(img1, img2);
end
