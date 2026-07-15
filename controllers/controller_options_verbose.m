function verbose = controller_options_verbose(options)
    % Return true unless an options struct explicitly disables constructor logs.
    verbose = true;
    if nargin >= 1 && isstruct(options) && isfield(options, 'verbose')
        verbose = logical(options.verbose);
    end
end
