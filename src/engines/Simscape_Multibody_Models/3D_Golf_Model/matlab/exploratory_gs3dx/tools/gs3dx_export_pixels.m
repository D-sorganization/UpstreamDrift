function pixels = gs3dx_export_pixels(fig, file, resolution)
%GS3DX_EXPORT_PIXELS Export offscreen graphics at explicit pixel dimensions without stretching.
    exportgraphics(fig,file,'Units','pixels','Width',resolution(1), ...
        'Height',resolution(2),'Padding','figure','PreserveAspectRatio','on','BackgroundColor','white');
    pixels=imread(file);
    % EXPORTGRAPHICS can round one dimension by a pixel. Adjust white border
    % pixels only; never rescale model geometry or crop nonwhite content.
    actual=[size(pixels,2),size(pixels,1)];
    assert(all(abs(actual-resolution)<=2),'gs3dx:render', ...
        'Native export rounding exceeded two pixels');
    if actual(2)>resolution(2)
        border=pixels(resolution(2)+1:end,:,:);
        assert(all(border==255,'all'),'gs3dx:render','Cannot crop nonwhite content');
        pixels=pixels(1:resolution(2),:,:);
    end
    if actual(1)>resolution(1)
        border=pixels(:,resolution(1)+1:end,:);
        assert(all(border==255,'all'),'gs3dx:render','Cannot crop nonwhite content');
        pixels=pixels(:,1:resolution(1),:);
    end
    if ~isequal([size(pixels,2),size(pixels,1)],resolution)
        canvas=uint8(255*ones(resolution(2),resolution(1),3));
        canvas(1:size(pixels,1),1:size(pixels,2),:)=pixels;
        pixels=canvas;
    end
    imwrite(pixels,file);
    assert(isequal([size(pixels,2),size(pixels,1)],resolution), ...
        'gs3dx:render','Exported dimensions differ from the requested resolution');
end



