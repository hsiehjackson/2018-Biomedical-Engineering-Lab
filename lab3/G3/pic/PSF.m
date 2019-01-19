clear all
DR = 60;	% dynamic range (動態範圍) of the image		
                % 因Sonosite無法顯示使用的動態範圍(通常會顯示在其他系統的原始影像上),可直接使用DR=60
Image1 = 'hw1_highgain_in.JPG';
Image2 = 'hw1_highgain_out.JPG';
Image3 = 'hw1_lowgain_in.JPG';
Image4 = 'hw1_lowgain_out.JPG';

Realx = 110:960; Realy = 280:980;
ptx_in =700:800; pty_in = 110:220; specx_in = 300:430; specy_in = 1:700;
% 在MATLAB中,imread進來的影像資料 data type 為 "uint8"
OriIm1 = imread(Image4);
%GrayIm = rgb2gray(OriIm);	% rgb to gray scale, data type : uint8

% 在MATLAB中，+,-,*,/等數值運算或函式只能使用於data type為double的資料上，
% 因此，在此先將uint8的data type轉成"double"
GrayIm1 = double(OriIm1);	
figure,imagesc(OriIm1), colormap(gray)

% 將原始影像上，真正屬於仿體影像的部份取出，不同的影像取的區域不同，
% 請自己找出自己擷取影像真正屬於仿體影像的部份
GrayIm1 = GrayIm1(Realx,Realy); 
OriIm1 = OriIm1(Realx,Realy);

% gray to dB 由0-255的灰階轉成 dB
dBIm1 = GrayIm1 - min(min(GrayIm1));	% set min value to 0
dBIm1 = dBIm1/max(max(dBIm1));			% normalization, 0 - 1
dBIm1 = dBIm1*DR;							% to dB, 0 - DR

% show B-mode image
figure;
subplot(2,5,[1 2,6 7]);
image(dBIm1)
colormap(gray(DR))
axis image
colorbar
title(strcat('B-mode image, dynamic range = ', num2str(DR), 'dB'))
hold on
rectangle('Position',[pty_in(1) ptx_in(1) pty_in(end)-pty_in(1) ptx_in(end)-ptx_in(1)],'EdgeColor','r')
hold on
rectangle('Position',[specy_in(1) specx_in(1) specy_in(end)-specy_in(1) specx_in(end)-specx_in(1)],'EdgeColor','g');

% ---------------------  estimate PSF size  -----------------
% 請比較不同深度上的點，PSF size有無差別?
% 比較single zone focusing與multi-zone focusing,同一深度的PSF size有無差別?

% 點的範圍
OriPt1 = OriIm1(ptx_in, pty_in);
ImPt1 = dBIm1(ptx_in, pty_in);    % 請輸入範圍

subplot(2,5,3);
imagesc(OriPt1),colormap(gray),axis image;

% 以lateral projection求橫向上的PSF size
ptLalProj = max(ImPt1) - max(max(ImPt1)); % normalise, in dB
x = 1:1:size(ptLalProj,2);
xq = 1:0.1:size(ptLalProj,2);
ptLalProj = interp1(x,ptLalProj,xq);
threshLal = ones(size(ptLalProj,2))*(-6);

% axial projection
ImPt1 = ImPt1.';
ptAxiProj = max(ImPt1) - max(max(ImPt1));   % 請transpose 點影像, 再取與lateral projection相同的計算方式
x = 1:1:size(ptAxiProj,2);
xq = 1:0.1:size(ptAxiProj,2);
ptAxiProj = interp1(ptAxiProj,xq);
threshAxi = ones(size(ptAxiProj,2))*(-6);

% 計算 lateral projection 的-6dB寬度. 此為psf size 
idxLal = find(ptLalProj >= -6 );	% find the indexes of the values, >= -6 dB
idxAxi = find(ptAxiProj >= -6 );
% !!!!!!!!!!! 近似求法，請以較準確的方法求出-6dB寬度?
Width6dBLal = (idxLal(end) - idxLal(1))*0.1; 
Width6dBAxi = (idxAxi(end) - idxAxi(1))*0.1;
                              % -6 dB width in index, !!! 此為近似的求法 (相當不準確)
							  % 因idx(end)與idx(1)所對應的值未必恰好等於 -6 dB，
                              % 要準確的求出-6dB寬度，至少得將對應-6 dB的index值內插出來再求
                              % 此外要注意，此時求出來的寬度僅為相差的index大小
                              % 若要換算成實際的單位如 mm，得找出兩個連續index間實際間隔多少mm
% show lateral projection
subplot(2,5,4);
plot(ptLalProj)
hold on
plot(threshLal,'r')
title(strcat('lateral projection, size = ', num2str(Width6dBLal)))

subplot(2,5,5);
plot(ptAxiProj)
hold on
plot(threshAxi,'r')
title(strcat('axial projection, size = ', num2str(Width6dBAxi)))

% 以axis projection求軸向上PSF size
% 幾乎與lateral projection一樣的求法


% ------------------ speckle std ------------------
% 計算speckle的標準差?
% 將影像上均質的部份選出來計算speckle std
% 可與理論值 4.34 dB做個驗證
spec_1 = dBIm1(specx_in, specy_in);
speckleStd = std2(spec_1);
subplot(2,5,8);
imagesc(spec_1),colormap(gray), axis image, set(gca, 'YTick', [0 25]), title(strcat('Standard Deviation = ', num2str(speckleStd), 'dB'));
                                          
% -------------------- speckle histogram -----------------------                                         
% dB to linear, 計算histogram前，得先將dB資料，轉為原來的linear的資料格式
% dB = 20*log10(E), E: amplitude => E = 10^(dB/20)
% dB = 10*log10(I), I: intensity => I = 10^(dB/10)
% 與計算std時一樣，取影像上均質的部份出來計算histogram
% 可使用MATLAB的函式"hist"來求出及畫出histgoram, "hist"僅只處理1-D的data,
% 所以需先使用reshape將2-D data 轉成1-D

% histogram 相當於 機率分布，(即高中所學的長條圖)
%可驗證speckle intensity及amplitude理論上的機率分布 (expenential distribution 及 Reyleigh distribution)

LinearIm_I = 10.^(spec_1/10);	% intensity 為 10.^(圖像dB值/10)
LinearIm_E = 10.^(spec_1/20);	% amplitude 為 10.^(圖像dB值/20)

subplot(2,5,9);
% 請先把 LinearIm_I 以及 LinearIm_E 變成column vector, 再使用hist指令畫出機率分布圖
hist(LinearIm_I(:),20);	% 像expenential distribution嗎?
title('Speckle Intensity Distribution');xlabel('I');ylabel('P_I')
subplot(2,5,10);
hist(LinearIm_E(:),20);	% 像Reyleigh distribution嗎?
title('Speckle Amplitude Distribution');xlabel('E');ylabel('P_E')