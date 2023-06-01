function [Tx,omega,STFT,time,freq] = MeinSST2_2(signal,fs,sigma,Fmax)
% Diese Programm ist Aligning Synchrosqueezing.
% Parameters:
%     Signal ist das signal. 信号  signal
%     fs ist die Abtastrate des Signal.   采频  sampling rate
%     sigma ist das Koeffizient der Gaussscher Fensterweite. 窗函数参数 sigma of gaussian window
%     hop ist die Verlagerung des Fenster.  窗移距离  time-shift
%     Fmax ist die Maximalwert der Frequenz.   最大频率坐标  maxvalue of the frequency axis
% 
%     Tx ist das Resultat. 时频图 TF representation    
%     omega isT der IF Operator. 瞬时频率估计算子 IF estimator
%     STFT ist der STFT. 短时傅里叶变换  STFT
%     time ist die Zeit. 时间轴  time axis
%     freq ist die Frequenz. 频率轴  frequency axis
%     
%     @Autor: Shaowen Chen
%     2020.12.15

[N,col] = size(signal);

if (col~=1)
%     error('Das Signal muss ein Spaltenvektor sein.');
     signal = signal';
     N = length(signal);
end

if (nargin < 2)
    error('Abtastrate muss eingegeben werden.');
end

if (nargin < 3)
    sigma = 0.008;%das Koeffizient der Fensterweite
    Fmax = fs/2; % die maximale Frequenz, die du m?chteest.(Nomaleweise ist Fmax fs/2.)
end

if (nargin < 4)
    Fmax = fs/2; % die maximale Frequenz, die du m?chteest.(Nomaleweise ist Fmax fs/2.)
end
gamma = 1e-8;
m = 1; 

LL = fs; % die Anzahl jedes FFT (Nomaleweise ist LL fs.)
% LL = 100;
freqLL = (0:LL-1)/LL*fs;
% Fmax = fs/2; % die maximale Frequenz, die du m?chteest.(Nomaleweise ist Fmax fs/2.)
aa = abs(freqLL-Fmax); Fmaxindex = find(aa == min(aa))-1;

sig_stft = hilbert(signal(:));

sleft =  zeros(LL/2,1);
sright = sleft;
sig_stft_ex = [sleft; sig_stft; sright];
sig_stft_ex = hilbert(real(sig_stft_ex));
fftindex = 1;

% Gaussscher Fenster funktion
w_halflen = LL/2;  
ix    = ((-w_halflen+0.5):(w_halflen-0.5));
t_win = ix(:)/fs;
% t_win = (1:60)/fs;
g     = (pi*sigma^2)^(-0.25).* exp(-(t_win/sigma).^2/2);
gp    = g.*(-t_win/sigma^2);

% Optionale Parameters
ft = 1:Fmaxindex;
ftt = 2:Fmaxindex;
bt = 1:m:N;

% Initialize TF storage space 
nb = length(bt); 
neta = length(ft);
STFT = zeros(neta,nb);
omega = zeros(neta,nb);
Tx = zeros(neta,nb);
IF = zeros(neta,nb);
% ser = zeros(neta,nb);

for u=1:nb
    df = freqLL(2)-freqLL(1);
    % In der Zeitpunkt t=b mit dem Fenster g STFT berechnen.
    tim = bt(u):bt(u)+LL-1;
    sig_stft_seg = sig_stft_ex(tim);
    tmp = fft(sig_stft_seg.*g)/LL*fftindex;

    % Werts speichern
    STFT(:,u) = tmp(ft);
    vtmp_g = tmp(ft);    
    
    % In der Zeitpunkt t=b mit dem Fenster g? STFT berechnen.   
    tmp = fft(sig_stft_seg.*gp)/LL*fftindex;
    vtmp_gp = tmp(ft);

    % In der Zeitpunkt t=b  Omega berechnen.
    omegtmp = freqLL(ft)'-imag(vtmp_gp(ft)./vtmp_g)/(2*pi);  % right  
    omegtmp3 = omegtmp;
    wk(ftt,u) =  abs(diff(omegtmp))/df;
    
    % Omega berichtigen
    index_sig = find(wk(:,u)<1);
%     ser(index_sig,nb) = 1 ;vtmp_g = vtmp_g.*ser(:,nb);
    N_mod = diff(index_sig);
    N_modd = find(N_mod >= 2);
    va = [1;N_mod(N_modd)];
    P_mod = [0;index_sig(N_modd);index_sig(end)];
    N_mod = length(P_mod) - 1;
    SEO = abs(-real(1i*vtmp_gp./vtmp_g/2/pi));
    for nn = 1:N_mod
        scope = P_mod(nn)+va(nn) : P_mod(nn+1);
        SEO_scope = SEO(scope);
        SEOmaxp = min(find(SEO_scope == min(SEO_scope))+P_mod(nn)+va(nn),neta);
        SEOmaxp = SEOmaxp(1);
        if omegtmp(SEOmaxp) ~= 0 
           omegtmp3(scope) = omegtmp(SEOmaxp)*ones(length(scope),1);
        end
    end
    
    omegtmp = omegtmp3;    
    omega(:,u) = omegtmp;
    
    % Wiederaufbau
    
    for eta=1:neta
        if abs(vtmp_g(eta))>gamma 
            k = 1+round((omegtmp(eta))/df);%  -ft(1)+1           
            if k>=1 && k<=neta
                Tx(k,u) = Tx(k,u) + vtmp_g(eta)*exp(1i*pi*(ft(eta)-1));
            end  
        end
    end
end

freq = freqLL(ft);
time = (0:(N-1))/fs;
end
