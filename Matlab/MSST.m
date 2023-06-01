function [Tx,omega,STFT,C] = MSST(signal,fs,sigma,Fmax)
%Diese Programm ist MSST.
%   fs ist die Abtastrate des Signal.
%   sigma ist das Koeffizient der Fensterweite.
[N,col] = size(signal);

if (col~=1)
%     error('Das Signal muss ein Spaltenvektor sein.');
     signal = signal';
     N = length(signal);
end

if (nargin < 2),
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
m = 1; nb = N;

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

% Gaussian Fenster funktion
w_halflen = LL/2;  
ix    = ((-w_halflen+0.5):(w_halflen-0.5));
t_win = ix(:)/fs;
% t_win = (1:60)/fs;
g     = (pi*sigma^2)^(-0.25).* exp(-(t_win/sigma).^2/2);
gp    = g.*(-t_win/sigma^2);
gpp   = g.*(t_win.^2/sigma^4) - g./sigma^2;
tg    = g.*t_win;
tgp   = gp.*t_win; 

% Optionale Parameters
ft = 1:Fmaxindex;
ftt = 2:Fmaxindex;
fttt = 1:Fmaxindex-1;
bt = 1:m:N;
time = bt/fs;

% Initialize TF storage space 
nb = length(bt); 
neta = length(ft);
STFT = zeros(neta,nb);
omega = zeros(neta,nb);
Tx = zeros(neta,nb);
C = zeros(neta,nb);

for u=1:nb
    df = freqLL(2)-freqLL(1);
    % In der Zeitpunkt t=b mit dem Fenster g STFT berechnen.
    tim = bt(u):bt(u)+LL-1;
    sig_stft_seg = sig_stft_ex(tim);
    tmp = fft(sig_stft_seg.*g)/LL*fftindex;

    % Werts speichern
    STFT(:,u) = tmp(ft);
    vtmp_g = tmp(ft);    
    vtmp_g1 = tmp(ft);
%     vtmp_g1((abs(vtmp_g)<gamma)) = NaN;   
    
    % In der Zeitpunkt t=b mit dem Fenster g¡¯ STFT berechnen.   
    tmp = fft(sig_stft_seg.*gp)/LL*fftindex;
    vtmp_gp = tmp(ft);

    % In der Zeitpunkt t=b  Omega berechnen.
    omegtmp = freqLL(ft)'-imag(vtmp_gp(ft)./vtmp_g)/(2*pi);  % right             
    
    % In der Zeitpunkt t=b mit dem Fenster g¡¯¡¯ STFT berechnen.    
    tmp = fft(sig_stft_seg.*gpp)/LL*fftindex;
    vtmp_gpp = tmp(ft);
    
    % In der Zeitpunkt t=b mit dem Fenster tg STFT berechnen.
    tmp = fft(sig_stft_seg.*tg)/LL*fftindex;
    vtmp_tg = tmp(ft);
    
    % In der Zeitpunkt t=b mit dem Fenster tg¡¯ STFT berechnen.    
    tmp = fft(sig_stft_seg.*tgp)/LL*fftindex;
    vtmp_tgp = tmp(ft);

    % In der Zeitpunkt t=b  Tau berechnen.    
    tautmp = u/fs + imag(1i*vtmp_tg./vtmp_g1);% right
    
    % C berechnen
    ctmp = imag((-vtmp_gp.^2+vtmp_gpp.*vtmp_g1)./(vtmp_tg.*vtmp_gp-vtmp_tgp.*vtmp_g1))/(2*pi);
    C(:,u) = ctmp;
    % Omega des MSST berechnen
    omegactmp = omegtmp + (ctmp.*(u/fs - tautmp));
    
    omega(:,u) = omegactmp;
    
    % Wiederaufbau
    
    for eta=1:neta
        if abs(vtmp_g(eta))>gamma 
            k = 1+round((omegactmp(eta))/df);%  -ft(1)+1           
            if k>=1 && k<=neta
                Tx(k,u) = Tx(k,u) + vtmp_g(eta)*exp(1i*pi*(ft(eta)-1));
            end  
        end
    end
end

end