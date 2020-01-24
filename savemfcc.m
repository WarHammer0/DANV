function savemfcc(file_name, save_dir)
        opt.fs = 16000;
        opt.Tw = 25;
        opt.Ts = 10;
        opt.alpha = 0.97;
        opt.R = [300 3700];
        opt.M = 13;
        opt.C = 13;
        opt.L = 22;
        % savemfcc('0572_0019_0003.wav','bin')
        [Speech, fs] = audioread(file_name);
        %Speech = Speech(1:400,:);
        [length_of_speech, channel] = size(Speech);
        if channel == 2
            Speech = (Speech(:, 1));
        end
        Speech = Speech(1:3440);
        
        [ MFCCs, ~, ~ ] = runmfcc( Speech, opt );
        mfccs = MFCCs(2:end, :);
        disp('output is:')
        disp(size(mfccs))
        num_bins = floor(length_of_speech / fs * 5);
        for l = 2:5:num_bins-1
            save_mfcc20 = mfccs(:, 20*l  : 20*l+19);
            pause
            f2 = fopen(fullfile(save_dir, [num2str(l), '.bin']), 'wb');
            
            fwrite(f2, save_mfcc20, 'double');
            fclose(f2);                    
        end
