clear all
DR = 60;	% dynamic range (�ʺA�d��) of the image		
                % �]Sonosite�L�k��ܨϥΪ��ʺA�d��(�q�`�|��ܦb��L�t�Ϊ���l�v���W),�i�����ϥ�DR=60
Image1 = 'hw1_highgain_in.JPG';
Image2 = 'hw1_highgain_out.JPG';
Image3 = 'hw1_lowgain_in.JPG';
Image4 = 'hw1_lowgain_out.JPG';

Realx = 110:960; Realy = 280:980;
ptx_in =700:800; pty_in = 110:220; specx_in = 300:430; specy_in = 1:700;
% �bMATLAB��,imread�i�Ӫ��v����� data type �� "uint8"
OriIm1 = imread(Image4);
%GrayIm = rgb2gray(OriIm);	% rgb to gray scale, data type : uint8

% �bMATLAB���A+,-,*,/���ƭȹB��Ψ禡�u��ϥΩ�data type��double����ƤW�A
% �]���A�b�����Nuint8��data type�ন"double"
GrayIm1 = double(OriIm1);	
figure,imagesc(OriIm1), colormap(gray)

% �N��l�v���W�A�u���ݩ����v�����������X�A���P���v�������ϰ줣�P�A
% �Цۤv��X�ۤv�^���v���u���ݩ����v��������
GrayIm1 = GrayIm1(Realx,Realy); 
OriIm1 = OriIm1(Realx,Realy);

% gray to dB ��0-255���Ƕ��ন dB
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
% �Ф�����P�`�פW���I�APSF size���L�t�O?
% ���single zone focusing�Pmulti-zone focusing,�P�@�`�ת�PSF size���L�t�O?

% �I���d��
OriPt1 = OriIm1(ptx_in, pty_in);
ImPt1 = dBIm1(ptx_in, pty_in);    % �п�J�d��

subplot(2,5,3);
imagesc(OriPt1),colormap(gray),axis image;

% �Hlateral projection�D��V�W��PSF size
ptLalProj = max(ImPt1) - max(max(ImPt1)); % normalise, in dB
x = 1:1:size(ptLalProj,2);
xq = 1:0.1:size(ptLalProj,2);
ptLalProj = interp1(x,ptLalProj,xq);
threshLal = ones(size(ptLalProj,2))*(-6);

% axial projection
ImPt1 = ImPt1.';
ptAxiProj = max(ImPt1) - max(max(ImPt1));   % ��transpose �I�v��, �A���Plateral projection�ۦP���p��覡
x = 1:1:size(ptAxiProj,2);
xq = 1:0.1:size(ptAxiProj,2);
ptAxiProj = interp1(ptAxiProj,xq);
threshAxi = ones(size(ptAxiProj,2))*(-6);

% �p�� lateral projection ��-6dB�e��. ����psf size 
idxLal = find(ptLalProj >= -6 );	% find the indexes of the values, >= -6 dB
idxAxi = find(ptAxiProj >= -6 );
% !!!!!!!!!!! ����D�k�A�ХH���ǽT����k�D�X-6dB�e��?
Width6dBLal = (idxLal(end) - idxLal(1))*0.1; 
Width6dBAxi = (idxAxi(end) - idxAxi(1))*0.1;
                              % -6 dB width in index, !!! ����������D�k (�۷��ǽT)
							  % �]idx(end)�Pidx(1)�ҹ������ȥ�����n���� -6 dB�A
                              % �n�ǽT���D�X-6dB�e�סA�ܤֱo�N����-6 dB��index�Ȥ����X�ӦA�D
                              % ���~�n�`�N�A���ɨD�X�Ӫ��e�׶Ȭ��ۮt��index�j�p
                              % �Y�n���⦨��ڪ����p mm�A�o��X��ӳs��index����ڶ��j�h��mm
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

% �Haxis projection�D�b�V�WPSF size
% �X�G�Plateral projection�@�˪��D�k


% ------------------ speckle std ------------------
% �p��speckle���зǮt?
% �N�v���W���誺������X�ӭp��speckle std
% �i�P�z�׭� 4.34 dB��������
spec_1 = dBIm1(specx_in, specy_in);
speckleStd = std2(spec_1);
subplot(2,5,8);
imagesc(spec_1),colormap(gray), axis image, set(gca, 'YTick', [0 25]), title(strcat('Standard Deviation = ', num2str(speckleStd), 'dB'));
                                          
% -------------------- speckle histogram -----------------------                                         
% dB to linear, �p��histogram�e�A�o���NdB��ơA�ର��Ӫ�linear����Ʈ榡
% dB = 20*log10(E), E: amplitude => E = 10^(dB/20)
% dB = 10*log10(I), I: intensity => I = 10^(dB/10)
% �P�p��std�ɤ@�ˡA���v���W���誺�����X�ӭp��histogram
% �i�ϥ�MATLAB���禡"hist"�ӨD�X�εe�Xhistgoram, "hist"�ȥu�B�z1-D��data,
% �ҥH�ݥ��ϥ�reshape�N2-D data �ন1-D

% histogram �۷�� ���v�����A(�Y�����ҾǪ�������)
%�i����speckle intensity��amplitude�z�פW�����v���� (expenential distribution �� Reyleigh distribution)

LinearIm_I = 10.^(spec_1/10);	% intensity �� 10.^(�Ϲ�dB��/10)
LinearIm_E = 10.^(spec_1/20);	% amplitude �� 10.^(�Ϲ�dB��/20)

subplot(2,5,9);
% �Х��� LinearIm_I �H�� LinearIm_E �ܦ�column vector, �A�ϥ�hist���O�e�X���v������
hist(LinearIm_I(:),20);	% ��expenential distribution��?
title('Speckle Intensity Distribution');xlabel('I');ylabel('P_I')
subplot(2,5,10);
hist(LinearIm_E(:),20);	% ��Reyleigh distribution��?
title('Speckle Amplitude Distribution');xlabel('E');ylabel('P_E')