function tests=test_gs3dx_upper_body_reference_binding
tests=functiontests(localfunctions);
end

function testlegacyParity(testCase)
[ik,jp,expected_angles]=fixture('GS3DX_Fit');
actual=gs3dx_upper_body_reference(ik);
verifyEqual(testCase,gs3dx_upper_body_reference(ik,joint_variables=jp),actual);
for k=1:numel(actual.joints)
 verifyEqual(testCase,actual.joints(k).angle,expected_angles{k},'AbsTol',1e-10);
 verifyEqual(testCase,actual.joints(k).rate,zeros(size(expected_angles{k})),'AbsTol',1e-10);
end
end

function testhumanRenumbering(testCase)
[fit,~]=fixture('GS3DX_Fit');[human,jp]=fixture('GS3DX_Human');
expected=gs3dx_upper_body_reference(fit);
actual=gs3dx_upper_body_reference(human,joint_variables=jp);
for k=1:numel(actual.joints)
 verifyEqual(testCase,actual.joints(k).angle,expected.joints(k).angle);
 verifyEqual(testCase,actual.joints(k).rate,expected.joints(k).rate);
 verifyNotEqual(testCase,actual.joints(k).ids,expected.joints(k).ids);
end
verifyEqual(testCase,actual.start,expected.start);
end

function testpermutationInvariance(testCase)
[ik,jp]=fixture('GS3DX_Human');
expected=gs3dx_upper_body_reference(ik,joint_variables=jp);
order=height(jp):-1:1;jp=jp(order,:);
ik.joint_ids=ik.joint_ids(order);ik.joint=ik.joint(order,:);
verifyEqual(testCase,gs3dx_upper_body_reference(ik,joint_variables=jp),expected);
end

function testhumanRequiresBinding(testCase)
[ik,~]=fixture('GS3DX_Human');
verifyError(testCase,@() gs3dx_upper_body_reference(ik),'gs3dx:ubref');
end

function testinvalidBindings(testCase)
[ik,jp]=fixture('GS3DX_Human');
bad=jp;bad.Unit(1)="rad";
verifyError(testCase,@() gs3dx_upper_body_reference(ik,joint_variables=bad),'gs3dx:ubref');
bad=jp;bad.ID(2)=bad.ID(1);
verifyError(testCase,@() gs3dx_upper_body_reference(ik,joint_variables=bad),'gs3dx:ubref');
bad=jp;bad.BlockPath(1)="OtherModel/foreign";
verifyError(testCase,@() gs3dx_upper_body_reference(ik,joint_variables=bad),'gs3dx:ubref');
bad=removevars(jp,'Unit');
verifyError(testCase,@() gs3dx_upper_body_reference(ik,joint_variables=bad),'gs3dx:ubref');
bad=jp(2:end,:);
verifyError(testCase,@() gs3dx_upper_body_reference(ik,joint_variables=bad),'gs3dx:ubref');
end

function testinvalidClockAndCoordinates(testCase)
[ik,jp]=fixture('GS3DX_Human');
bad=ik;bad.t(3)=bad.t(2);
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
bad=ik;bad.t(3)=bad.t(3)+.001;
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
bad=ik;bad.joint(1,1)=NaN;
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
bad=ik;bad.joint_ids{2}=bad.joint_ids{1};
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
end

function testmalformedFramesFailClosed(testCase)
[ik,jp]=fixture('GS3DX_Human');
bad=ik;bad.frames=1:0.5:16.5;
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
bad=ik;bad.status(2)=0;
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
bad=ik;bad.joint=bad.joint(:,1:end-1);
verifyError(testCase,@() gs3dx_upper_body_reference(bad,joint_variables=jp),'gs3dx:ubref');
end

function testUnfilteredGeometryIsPreserved(testCase)
[ik,jp,expected_angles]=fixture('GS3DX_Human');spec=gs3dx_upper_body_joints();
wave=sin(2*pi*20*ik.t);
for k=1:numel(spec)
 rows=find(string(jp.BlockPath)==string(ik.model)+"/"+spec(k).block);
 if numel(spec(k).axes)==3
  ik.joint(rows(4),:)=ik.joint(rows(4),:)+wave;
  expected_angles{k}(1,:)=expected_angles{k}(1,:)+wave;
 else
  ik.joint(rows,:)=ik.joint(rows,:)+wave;
  expected_angles{k}=expected_angles{k}+wave;
 end
end
actual=gs3dx_upper_body_reference(ik,joint_variables=jp,filter_reference=false);
filtered=gs3dx_upper_body_reference(ik,joint_variables=jp);
verifyFalse(testCase,actual.filter_applied);verifyTrue(testCase,filtered.filter_applied);
for k=1:numel(spec)
 verifyEqual(testCase,actual.joints(k).angle,expected_angles{k},'AbsTol',1e-10);
 verifyTrue(testCase,all(isfinite(actual.joints(k).rate),'all'));
end
verifyGreaterThan(testCase,max(abs(actual.joints(1).angle-filtered.joints(1).angle),[],'all'),1e-3);
end

function [ik,jp,expected_angles]=fixture(model)
spec=gs3dx_upper_body_joints();t=(0:31)/60;
ids=strings(0,1);paths=ids;units=ids;q=zeros(0,numel(t));expected_angles=cell(1,numel(spec));
for k=1:numel(spec)
 j=spec(k);stem=string(j.id);
 if strcmp(model,'GS3DX_Human'),stem="j"+string(k+30);end
 switch numel(j.axes)
  case 0,suffix="Rz.q";
  case 2,suffix=["Rx.q";"Ry.q"];
  otherwise,suffix=["S.ax_x";"S.ax_y";"S.ax_z";"S.q"];
 end
 values=repmat(k+(1:numel(suffix)).',1,numel(t));
 expected_angles{k}=values;
 native_units=repmat("deg",numel(suffix),1);
 if numel(j.axes)==3
  values=[ones(1,numel(t));zeros(2,numel(t));repmat(k,1,numel(t))];
  expected_angles{k}=[repmat(k,1,numel(t));zeros(2,numel(t))];
  native_units(1:3)="1";
 end
 ids=[ids;stem+"."+suffix];paths=[paths;repmat(string(model)+"/"+j.block,numel(suffix),1)]; %#ok<AGROW>
 units=[units;native_units];q=[q;values]; %#ok<AGROW>
end
jp=table(ids,paths,units,'VariableNames',{'ID','BlockPath','Unit'});
ik=struct('model',model,'frames',1:32,'t',t,'status',ones(1,32),'joint_ids',{cellstr(ids)},'joint',q);
end
