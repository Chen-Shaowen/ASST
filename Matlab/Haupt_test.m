clc;
close all
clear all;
% randn('state',1)
%%
%信号生成与输出
fs=1024;
t=1/fs:1/fs:1; 
%信号
t1=t(1:fs/2);
t2=t((fs/2+1):fs);
signal7 = [cos(400*pi*t1-300*pi*t1.^2) cos(400*pi*t2-600*pi*t2.^2+400*pi*t2.^3+pi)];
signal8 = cos(2*pi*(350*t+40*sin(8*pi*t.^2)/(8*pi)));
% signal8 = cos(2*pi*(350*t+45*sin(15*pi*t.^2)/(15*pi)));
Amp1 = (1.5-0.1*cos(2*pi*t));
Amp2 = (1+0.6*t);
signal7 = Amp1.*signal7;signal8 = Amp2.*signal8;
signal = signal7 + signal8;

%% bat signal
% % load ('batsignal.mat')
% load ('signal_SNR.mat')
% % signal = batsignal';
% % t=1/fs:1/fs:400/fs; 
% % signall = signal(13,:);
% % signal = signal(12,:);
% load('Haupt_signal_3.mat')
% load('signal_for_TUM.mat')
%%
frequency1 = [(200-300*t1) (200-600*t2+600*t2.^2)];
frequency2 = 350+80*t.*cos(8*pi*t.^2);

%噪声
% signal=awgn(signal,30);
freqN = fs/2*linspace(0,1,fs/2); %频率坐标

figure('NumberTitle','off','Name','原信号')
plot(t,signal,'b')
ylabel('Amp / mm');
xlabel('Time / Sec');
title('原信号')

figure('NumberTitle','off','Name','原信号频率')
plot(t,frequency1,'r');hold on;
plot(t,frequency2,'r');
ylabel('Freq / Hz');
xlabel('Time / Sec');
ylim([1,fs/2]);

LL = fs; % die Anzahl jedes FFT (Nomaleweise ist LL fs.)
% LL = 100;
freqLL = (0:LL-1)/LL*fs;
Fmax = fs/2; % die maximale Frequenz, die du m?chteest.(Nomaleweise ist Fmax fs/2.)
aa = abs(freqLL-Fmax); Fmaxindex = find(aa == min(aa))-1;

nb = length(signal);neta = nb/2;
sigma = 0.008;
%%
% 
% [Tx3,omega3,STFT] = SST(signal(11,:)',fs);
% [Tx4,omega4,STFT] = SST(signal(12,:)',fs);
% [Tx5,omega5,STFT] = SST(signal(13,:)',fs);
[Tx1,omega1,STFT] = SST(signal',fs);
% [Tx3,omega3] = MeinSST2_2(signal',fs);
[Tx4] = SET_Y(signal',fs);
[Tx5,omega5,~,C] = MSST(signal',fs);
% [Tx5,omega5] = MeinSST2_2_fast(signal',fs,0.008,1);
% [Tx1,Tx3,Tx4,Tx5,omega1,omega3,omega4,omega5,STFT] = SST_N(signal',fs);
% [~,~,~,Tx5,~,~,~,omega5,STFT] = SST_N(signal',fs);
% [Tx3,Tx31,omega3,omega31] = ASST_with_statistical_denoising(signal',fs);
[Tx3,Tx31,omega3,omega31] = MeinSST2_2(signal',fs);

clr = 128;
figure(3);set(gcf, 'NumberTitle','off','Name','STFT');
TFimage = (abs(STFT(1:Fmaxindex,:)));
imagesc(t,freqLL(1:Fmaxindex),TFimage);
colormap(jet_Linchao(clr))
axis xy;
xlim_min = min(t);  xlim_max = max(t); 
ylim_min = freqLL(1);  ylim_max = freqLL(Fmaxindex);
xylim = [xlim_min,xlim_max,ylim_min,ylim_max]; axis(xylim);
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('STFT');
% xlim([0.26,0.45]);
% ylim([300,465]);


figure(4);set(gcf, 'NumberTitle','off','Name','Omega');
imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(omega1));
colormap(jet_Linchao(clr))
axis xy
caxis ([0 512])
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('Omega');

% figure(6);set(gcf, 'NumberTitle','off','Name','Corrected Omega by Assignment');
% imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(omega3));
% colormap(jet_Linchao(clr))
% axis xy
% caxis ([0 512])
% ylabel('Freq / Hz');
% xlabel('Time / Sec');
% title('Corrected Omega by Assignment');

figure(7);set(gcf, 'NumberTitle','off','Name','Omega of MSST');
imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(omega5));
colormap(jet_Linchao(clr))
axis xy
caxis ([0 512])
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('Omega of MSST');

figure(8);set(gcf, 'NumberTitle','off','Name','SST');
imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(Tx1));
colormap(jet_Linchao(clr))
axis xy
caxis auto
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('SST');
% xlim([0.26,0.45]);
% ylim([300,465]);

figure(10);set(gcf, 'NumberTitle','off','Name','ASST');
imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(Tx3));
colormap(jet_Linchao(clr))
axis xy
caxis auto
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('ASST');
% xlim([0.26,0.45]);
% ylim([300,465]);
% hold on;plot(t,frequency1,'b');hold on;
% plot(t,frequency2,'b');

figure(11);set(gcf, 'NumberTitle','off','Name','SET');
imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(Tx4));
colormap(jet_Linchao(clr))
axis xy
caxis auto
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('SET');
% xlim([0.26,0.45]);
% ylim([300,465]);

figure(12);set(gcf, 'NumberTitle','off','Name','MSST');
imagesc((1:nb)/fs,fs/2*linspace(0,1,neta),abs(Tx5));
colormap(jet_Linchao(clr))
axis xy
caxis auto
ylabel('Freq / Hz');
xlabel('Time / Sec');
title('MSST');