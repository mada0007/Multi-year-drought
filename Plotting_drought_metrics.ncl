;================================================================================================================================
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin
;================================================================================================================================
; variable and file handling
;================================================================================================================================


duration_spi1 = addfile("/g/data/w35/ma9839/awap_drought/AWAP_SPI1.nc/SPI_moderate_scale_1/AWAP_SPI1.nc", "r")
duration_spi2 = addfile("/g/data/w35/ma9839/awap_drought/SPI3AWAP_SPI3.nc/SPI_moderate_scale_3/AWAP_SPI3.nc", "r")
 
;================================================================
;================================================================
; ********************** Historical ***************************** 
x1=dim_max_n_Wrap(duration_spi1->duration(:,:,:), (/0/))
x2=dim_max_n_Wrap(duration_spi2->duration(:,:,:), (/0/))

print(x2)



 

;================================================================================================================================
;================================================================================================================================
;================================================================================================================================
;===========================================================================================================================================================================



;=====================================================================================================
        wks  = gsn_open_wks("png","/g/data/w35/ma9839/FOR_AILIE/Results/DURATION_SPI1_SPI3_v3_latest")            
        plot = new(2,graphic)                        
  	res                      = True               
  	res@gsnFrame = False 
 	res@gsnDraw = False
  	res@cnFillOn             = True             
  	;res@cnFillPalette        = "NCV_blu_red" 
	; levels = ispan(50,70,1)
    ; icol = get_color_index("MPL_gist_earth",levels,0.5)     

	color = read_colormap_file("MPL_rainbow")
	print(color)
	color_r = color(::-1,:) 
	res@cnFillPalette        = color
  	res@gsnAddCyclic         = False
  	res@cnLinesOn            = False  
  	res@cnLineLabelsOn = False
  	res@cnLinesOn            = False             
  	res@cnLevelSelectionMode = "ManualLevels"     
  	res@mpFillDrawOrder = "PostDraw"
	res@mpLimitMode = "latlon"
	res@mpMinLatF = -45
 	res@mpMaxLatF = -10.0
 	res@mpMinLonF = 110.0
 	res@mpMaxLonF = 160.0
	res@mpCenterLonF = (res@mpMaxLonF + res@mpMinLonF)*0.5
	res@mpFillDrawOrder = "postdraw"
	res@mpLandFillColor = (/1.0,1.0,1.0,0.0/) ; r,g,b,opacity
	res@mpOceanFillColor = (/1.0,1.0,1.0,1.0/) ; r,g,b,opacity
  res@lbLabelBarOn         = False 

  res@gsnStringFontHeightF = 0.019
  res@tmXBLabelFontHeightF = 0.019             
  res@tmYLLabelFontHeightF = 0.019
;=====================================================================================================
;=====================================================================================================
	res@cnMinLevelValF       = 0               ; min level
   
  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
;   res@cnLevels    = (/0,3,6,9,12,15,18,21,24,30/)   ; set levels            ; contour level spacing
  res@cnLevels    = (/0,1,2,3,4,5,6,7,8,9,10,11,12,13,14/)   ; set levels            ; contour level spacing

 	res@gsnLeftString		= "(a)"
 	res@gsnRightString 		= 	"SPI-1"
  
 	plot(0) = gsn_csm_contour_map(wks,x1,res)

  res@gsnLeftString		= "(b)"	
  res@gsnRightString 		= "SPI-3"
  plot(1) = gsn_csm_contour_map(wks,x2,res)

;================================================================
;================================================================

sig_res =True
sig_res@gsnDraw =False
sig_res@gsnFrame =False
sig_res@cnFillOn = True
sig_res@cnLinesOn =False
sig_res@cnFillPattern =17
sig_res@cnFillDotSizeF =0.0025
sig_res@cnMonoFillColor =True
sig_res@cnFillColor ="black"
sig_res@lbLabelBarOn =False
sig_res@cnInfoLabelOn =False
sig_res@cnLineLabelsOn =False

;=================================================================================================================
   res_P             = True                      
   res_P@gsnPanelLabelBar= True   
   res_P@lbLabelFontHeightF =.015 
   res_P@lbTitleString    = "months"
    res_P@lbAutoManage         = False

   res_P@lbTitlePosition = "bottom"
   res_P@lbBoxEndCapStyle = "TriangleHighEnd"
   res_P@lbTitleFontHeightF= .015 
   gsn_panel(wks,plot,(/1,2/),res_P)             

end

	