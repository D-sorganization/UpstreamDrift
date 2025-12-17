function str = logical2str(logical_val)
    if logical_val
        str = 'enabled';
    else
        str = 'disabled';
    end
end
