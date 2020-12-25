;================================================================================================================================
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin
;================================================================================================================================
; variable and file handling
;================================================================================================================================



duration_spi1 = addfile("/g/data/w35/ma9839/AWAP_SPI1.nc/SPI_moderate_scale_1/AWAP_SPI1.nc", "r")  ; import date for drought metrics spi < +1
duration_spi2 = addfile("/g/data/w35/ma9839/FOR_AILIE/years_all_actual.nc","r")
print(duration_spi1)
 
;================================================================
;================================================================
; ********************** Historical ***************************** 
x1=dim_max_n_Wrap(duration_spi1->duration(:,:,:), (/0/))
x2=(duration_spi2->years)


copy_VarCoords(x1, x2)

printVarSummary(x1)
printVarSummary(x2)
print(x2)




 

;================================================================================================================================
;================================================================================================================================
;================================================================================================================================
;===========================================================================================================================================================================



;=====================================================================================================
        wks  = gsn_open_wks("png","/g/data/w35/ma9839/FOR_AILIE/Results/MAX_DROUGHT_DURATION_SPI<+1_and_years_of_MAX_2")            
        plot = new(2,graphic)                        
  	res                      = True               
  	res@gsnFrame = False 
 	res@gsnDraw = False
  	res@cnFillOn             = True             
    
	color1 = read_colormap_file("rainbow")   ; color for years plot
	; color = read_colormap_file("temp_19lev")
	color = read_colormap_file("MPL_rainbow")


	res@cnFillPalette        = color ;selecting color range
	; res@cnFillPalette        = color  ;all colors
  	res@gsnAddCyclic         = False
  	res@cnLinesOn            = True 
  	res@cnLineLabelsOn = False
  	res@cnLinesOn            = False 
	res@cnFillMode = "RasterFill" 
	 res@cnRasterSmoothingOn = True    ;smoothening array::      
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
    res@lbLabelBarOn         = True 

  res@gsnStringFontHeightF = 0.019
  res@tmXBLabelFontHeightF = 0.019             
  res@tmYLLabelFontHeightF = 0.019
  res@lbLabelFontHeightF =.016 
  
  

;=====================================================================================================
;=====================================================================================================
 	
   
  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
 	res@gsnLeftString		= "(a)"
 	res@gsnRightString 		= 	"(SPI < +1)"
    res@cnLevels            = ispan(20,60,2)
	res@lbLabelFontHeightF =.017 
	res@lbTopMarginF         = 0.16 ; increase space between top margin and label bar

	res@lbTitleString    = "months"
	

   res@lbTitlePosition = "bottom"
   res@lbBoxEndCapStyle = "TriangleBothEnds"


 	plot(0) = gsn_csm_contour_map(wks,x1,res)

    res@cnLevels            := ispan(1911,2014,5)
    res@cnFillPalette       := color          ; contour level spacing
    res@gsnRightString = ""
    res@gsnLeftString		= "(b)"	
	res@lbTitleString    = "years"

   res@lbTitlePosition = "bottom"
   res@lbBoxEndCapStyle = "RectangleEnds"

  plot(1) = gsn_csm_contour_map(wks,x2,res)




;=====================================================================================
;=================================================================================================================
   res_P             = True                      
   ;res_P@gsnMaximize = True                       
;    res_P@gsnPanelLabelBar= True   
;    res_P@lbLabelFontHeightF =.015 
;   res_P@gsnMaximize        = True 
   gsn_panel(wks,plot,(/1,2/),res_P)             

end

	