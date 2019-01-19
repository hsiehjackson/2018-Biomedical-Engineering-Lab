%time =  table2array(EEG(3994:5994,1));
%alpha = table2array(EEG(3994:5994,3));
%beta = table2array(EEG(3994:5994,4));
%delta = table2array(EEG(1994:3994,5));
%theta = table2array(EEG(1994:3994,6));

time = [1:2225];
wave1 = table2array(w10_1(1:2225,15));
wave2 = table2array(w10_1(1:2225,16));
wave3 = table2array(w8_1(1:2225,15));
wave4 = table2array(w8_1(1:2225,16));
subplot(4,1,1);
plot(time,wave1);
subplot(4,1,2);
plot(time,wave2);
subplot(4,1,3);
plot(time,wave3);
subplot(4,1,4);
plot(time,wave4);
%Fs = 200;
%alpha = alpha-mean(alpha);
%wave = alpha;
%y = detrend(wave);
%Y = fft(y,length(y));

%amp = abs(Y);
%n = 0:(length(y)-1);
%f = n*Fs/length(y);
%subplot(1,1,1);
%plot(time,wave);
%ylim([-6,6]);
%subplot(2,1,2)
%plot(f(1:length(y)/20),amp(1:length(y)/20))




