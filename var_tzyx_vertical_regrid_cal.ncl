begin

   filename = "temp_var.nc"

; Read data for lon and lat and output
   in  = addfile(filename,"r")
   P0 = in->P0
   hyam = in->hyam
   hybm = in->hybm
   var_tmp1 = in->var_tmp1
   ps = in->ps

   dim3 = dimsizes(var_tmp1)

;   print(dim3)        
;   print(hyam)

   var_out = new((/19, dim3(1), dim3(2)/), typeof(var_tmp1))

   pnew = (/1000,925,850,700,600,500,400,300,250,200,150,100,70,50,30,20,10,5,1/) 

; convert grid
   var_out(:,:,:) = vinth2p(var_tmp1(:,:,:),hyam,hybm,pnew,ps(:,:),2,P0,1,False)

; Output file
   filename_out = "output_from_ncl.nc"
   system("/bin/rm -f " + filename_out)
   fout  = addfile (filename_out, "c")  ; open output file
   fout->lev = (/pnew/) 
   fout->var_regrid = (/var_out/)

end
