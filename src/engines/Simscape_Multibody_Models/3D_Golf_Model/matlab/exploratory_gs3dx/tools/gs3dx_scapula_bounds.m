function b = gs3dx_scapula_bounds(jp, layout, address_deg)
%GS3DX_SCAPULA_BOUNDS  Bilateral Protraction Bounds in Independent IK Radians.
% ADDRESS_DEG == 0 disables the policy for exact legacy solver behavior.
% Otherwise address is confined to 5..10 degrees and the remaining swing
% to 5..the registry protraction limit (currently 25 degrees).
% This is a requested matching prior, not measured anatomy.
% Signs and neutral coordinates come from the existing joint ROM registry.
% Native IDs are resolved by block roles, including renumbered variants.
    assert(isnumeric(address_deg) && isscalar(address_deg) && isreal(address_deg) && ...
        isfinite(address_deg) && (address_deg == 0 || (address_deg >= 5 && address_deg <= 10)), ...
        'gs3dx:ik:scapula', 'Address protraction must be 0 (disabled) or 5..10 degrees');
    address_deg=double(address_deg);
    b=struct('active',address_deg>0);
    if ~b.active, return; end
    assert(istable(jp) && all(ismember({'ID','BlockPath','Unit'},jp.Properties.VariableNames)), ...
        'gs3dx:ik:scapula','Joint table must include ID, BlockPath and Unit');
    n=sum([layout.n]);b.lower=-inf(n,1);b.upper=inf(n,1);
    b.address_lower=b.lower;b.address_upper=b.upper;b.target=zeros(n,1);
    b.indices=zeros(2,1);b.sign=zeros(2,1);b.neutral_deg=zeros(2,1);
    registry=gs3dx_joint_rom();
    starts=[0,cumsum([layout.n])];keys=string({layout.key});
    for side=1:2
        role=["Left Scapula","Right Scapula"];joint=["LScap","RScap"];
        m=contains(string(jp.BlockPath),role(side)) & endsWith(string(jp.ID),'.Rx.q');
        assert(nnz(m)==1 && string(jp.Unit(m))=="deg", ...
            'gs3dx:ik:scapula','Expected one native-degree protraction Rx for each scapula');
        key=extractBefore(string(jp.ID(m)),'.q');k=find(keys==key);
        assert(numel(k)==1 && layout(k).n==1,'gs3dx:ik:scapula', ...
            'Scapula protraction must be an independent scalar coordinate');
        row=registry.joint==joint(side) & endsWith(registry.key,'|Rx.q');
        assert(nnz(row)==1,'gs3dx:ik:scapula','Missing scapula ROM convention');
        sign=registry.sign(row);neutral=registry.neutral_deg(row);idx=starts(k)+1;
        assert(ismember(sign,[-1,1]) && isfinite(neutral),'gs3dx:ik:scapula','Invalid scapula convention');
        limit=registry.max_deg(row);
        assert(isfinite(limit) && limit>=10,'gs3dx:ik:scapula','Invalid scapula protraction limit');
        swing=deg2rad(neutral+sign*[5,limit]);
        address=deg2rad(neutral+sign*[5,10]);
        b.lower(idx)=min(swing);b.upper(idx)=max(swing);
        b.address_lower(idx)=min(address);b.address_upper(idx)=max(address);
        b.target(idx)=deg2rad(neutral+sign*address_deg);
        b.indices(side)=idx;b.sign(side)=sign;b.neutral_deg(side)=neutral;
    end
end
