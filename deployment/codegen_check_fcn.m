function codegen_check_fcn(fname)
%**************************************************************************
% 上机前的代码生成检查：按 MATLAB Function 块的约束（标量 double 输入，个数按函数定义）
% 对上机版 fcn 生成 C 代码，网络用不依赖第三方库的通用实现。
% 目的是在实验室编译之前就发现不支持的函数、类型不一致等错误（8/25 现场编译报错的教训）。
%   codegen_check_fcn('fcn_S8_OLc')
%**************************************************************************
if nargin < 1, fname = 'fcn_S8_OLc'; end
cd(fileparts(mfilename('fullpath')));
out = fullfile(tempdir, ['cg_' fname]);
for lang = {'C', 'C++'}
    cfg = coder.config('lib'); cfg.GenCodeOnly = true; cfg.TargetLang = lang{1};
    cfg.DeepLearningConfig = coder.DeepLearningConfig('none');
    try
        codegen('-config', cfg, fname, '-args', num2cell(zeros(1, nargin(fname))), '-d', out);   % 输入个数按函数定义（4 或 5 个标量）
        fprintf('代码生成通过：%s（%s）\n', fname, lang{1}); return
    catch err
        fprintf('代码生成（%s）失败：%s\n', lang{1}, err.message);
    end
end
error('代码生成两种语言都失败：%s', fname);
end
