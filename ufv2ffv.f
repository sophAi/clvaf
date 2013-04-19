* -----------------------------------------------------------------------
*  sophAi - Statistical Optimization & Phenomenonal Histogram Ai
*
*  http://github.com/sophAi/sophAi
*  Written by Po-Jen Hsu, xanadu8850@gmail.com
*  Copyright (c),2012-, sophAi and Po-Jen Hsu.
*
*  This software is distributed under the GNU General Public License.
*  See the README file in the top-level sophAi directory.
* -----------------------------------------------------------------------
*  | File Name : ufv2ffv.f
*  | Creation Time : 2013-04-16 21:37:24
*  | Last Modified : 2013-04-18 10:03:01
* -----------------------------------------------------------------------
      program ufv2ffv
      implicit none
      integer I0,I1,I2,I3
      integer header_par_num,ndim_fac,atom_num,file_x_dim,
     &file_y_dim
      character header_source_type*3
      character dummy, read_file_name*80, write_file_name*80
      character header_par_name(100)*25,header_annotation*80
      real*8 header_par_real(100),v(3000)
      write(*,*) "Convert ufv file generated by sophAi to a "
      write(*,*) "formatted (ASCII) ffv file that can be read by"
      write(*,*) "CLTCF"
      write(*,*)
      write(*,*) "Please input your source ufv file name (ex: *.ufv)"
      read(*,*) read_file_name
      write(*,*) "Please input your output ffv file name (ex: *.ffv)"
      read(*,*) write_file_name
      open(21,file=read_file_name,form="unformatted",status="old")
      open(22,file=write_file_name,status="replace")
      read(21) dummy, header_source_type,header_par_num,
     &(header_par_name(I0),header_par_real(I0),I0=1,header_par_num),
     &header_annotation
      write(*,*) "Header information: "
      write(*,*) "Header source type: ",header_source_type
      write(*,*) "Header annotation: ",header_annotation
      write(*,*) "Total header parameters: ",header_par_num
      do I0=1,header_par_num
        write(*,*) "Header= ",I0,", name= ",header_par_name(I0),
     &", value= ",header_par_real(I0)
        if(header_par_name(I0).eq."file_y_dim")then
          file_y_dim=dint(header_par_real(I0))
        endif
        if(header_par_name(I0).eq."ndim_fac")then
          ndim_fac=dint(header_par_real(I0))
        endif
        if(header_par_name(I0).eq."atom_num")then
          atom_num=dint(header_par_real(I0))
        endif
        if(header_par_name(I0).eq."file_x_dim")then
          file_x_dim=dint(header_par_real(I0))
          write(*,*) "How many frames would you like to copy?(must <= ",
     &file_x_dim,")"
          read(*,*)file_x_dim
          header_par_real(I0)=dble(file_x_dim)
        endif
      enddo
      write(*,*) "Converting file..."
      write(*,*) "There are ",atom_num," atom(s)."
      write(22,*)dummy," ffv ", header_par_num, " ",
     &(header_par_name(I0),header_par_real(I0),I0=1,header_par_num)," ",
     &header_annotation
      rewind(21)
      do I0=0,atom_num-1
        read(21)
        do I1=0,file_x_dim-1
          read(21) (v(I2),I2=1,atom_num*ndim_fac)
          write(22,*) (v(I0*ndim_fac+I3), I3=1,ndim_fac)
        enddo
        write(*,*) "Atom ",I0+1," complete!"
        rewind(21)
      enddo
      write(*,*) "Complete!"
      close(21)
      close(22)
      stop
      end
