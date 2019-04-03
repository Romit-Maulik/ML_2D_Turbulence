!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!
!This set of subroutines utilized for ML based approximate deconvolution
!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!


!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!
!Subroutine to calculate the mean of an intent in field
!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!
subroutine mean_calc(w,nx,ny,mean)
	implicit none

	integer, intent(in) :: nx,ny
	double precision, intent(in),dimension(0:nx-1,0:ny-1) :: w
	double precision, intent(out) :: mean

	integer :: i, j

	mean = 0.0d0

	do j = 0,ny-1
		do i = 0,nx-1
			mean = mean + w(i,j)
		end do
	end do

	mean = mean/dfloat(nx*ny)


	return
end

!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!
!Subroutine to calculate the standard deviation of an intent in field
!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!
subroutine std_calc(w,nx,ny,sdev)
	implicit none

	integer, intent(in) :: nx,ny
	double precision, intent(in),dimension(0:nx-1,0:ny-1) :: w
	double precision, intent(out) :: sdev

	integer :: i, j
	double precision :: mean

	mean = 0.0d0

	do j = 0,ny-1
		do i = 0,nx-1
			mean = mean + w(i,j)
		end do
	end do

	mean = mean/dfloat(nx*ny)

	sdev = 0.0d0

	do j = 0,ny-1
		do i = 0,nx-1
			sdev = sdev + (w(i,j)-mean)**2
		end do
	end do

	sdev = dsqrt(sdev/(dfloat(nx*ny-1)))

	return
end

!---------------------------------------------------------------------------------------!
!---------------------------------------------------------------------------------------!
!Subroutine to calculate jacobian of intent in omega and psi fields
!---------------------------------------------------------------------------------------!
!Validated
!-------------------------------------------------------------------------!
subroutine jacobian_calc(omega,psi,jc,nx,ny,dx,dy)
!$ use omp_lib
implicit none

integer,intent(in) :: nx, ny
integer :: i,j

double precision, intent(in) :: dx, dy

double precision,intent(in),dimension(0:nx-1,0:ny-1) :: omega, psi
double precision,intent(out),dimension(0:nx-1,0:ny-1) :: jc

double precision, dimension(:,:), allocatable :: psi_new, omega_new
double precision :: jj1, jj2, jj3

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))

do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do

call periodic_bc_update(nx,ny,psi_new)
call periodic_bc_update(nx,ny,omega_new)

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1


jj1 = 1.0/(4.0*dx*dy) * ((omega_new(i+1,j)-omega_new(i-1,j)) * (psi_new(i,j+1) - psi_new(i,j-1)) &
			- (omega_new(i,j+1)-omega_new(i,j-1)) * (psi_new(i+1,j) - psi_new(i-1,j)))

jj2 = 1.0 / (4.0 * dx * dy) * (omega_new(i+1, j) * (psi_new(i+1, j+1) - psi_new(i+1, j-1)) &
                                         - omega_new(i-1, j) * (psi_new(i-1, j+1) - psi_new(i-1, j-1)) &
                                         - omega_new(i, j+1) * (psi_new(i+1, j+1) - psi_new(i-1, j+1)) &
                                         + omega_new(i, j-1) * (psi_new(i+1, j-1) - psi_new(i-1, j-1)) &
                                          )

jj3 = 1.0 / (4.0 * dx * dy) * (omega_new(i+1, j+1) * (psi_new(i, j+1) - psi_new(i+1, j)) &
                                        -  omega_new(i-1, j-1) * (psi_new(i-1, j) - psi_new(i, j-1)) &
                                        -  omega_new(i-1, j+1) * (psi_new(i, j+1) - psi_new(i-1, j)) &
                                        +  omega_new(i+1, j-1) * (psi_new(i+1, j) - psi_new(i, j-1)) &
                                          )

jc(i, j) = (jj1 + jj2 + jj3)/3.0

end do
end do
!$OMP END PARALLEL DO

deallocate(psi_new,omega_new)

return
end

!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!Validated
!---------------------------------------------------------------------------!
subroutine periodic_bc_update(nx,ny,u)
implicit none

integer :: nx, ny, i, j
double precision, dimension(-1:nx,-1:ny) :: u

do i = 0,nx-1
u(i,-1) = u(i,ny-1)
u(i,ny) = u(i,0)
end do

do j = -1,ny
u(-1,j) = u(nx-1,j)
u(nx,j) = u(0,j)
end do

end subroutine

!-----------------------------------------------------------------------------------!
!-----------------------------------------------------------------------------------!
!subroutine to denormalize a field from zero mean and unit sdev
!-----------------------------------------------------------------------------------!
!-----------------------------------------------------------------------------------!
subroutine denormalize_field(w,nx,ny,mean,sdev)
	implicit none

	integer, intent(in) :: nx, ny
	integer :: i, j

	double precision, intent(in) :: mean, sdev
	double precision, intent(inout), dimension(0:nx-1,0:ny-1) :: w


	do j= 0,ny-1
		do i = 0,nx-1
			w(i,j) = w(i,j)*sdev + mean
		end do
	end do


	return
end


!-----------------------------------------------------------------------------------!
!-----------------------------------------------------------------------------------!
!subroutine to normalize a field from zero mean and unit sdev
!-----------------------------------------------------------------------------------!
!-----------------------------------------------------------------------------------!
subroutine normalize_field(w,nx,ny,mean,sdev)
	implicit none

	integer, intent(in) :: nx, ny
	integer :: i, j

	double precision, intent(in) :: mean, sdev
	double precision, intent(inout), dimension(0:nx-1,0:ny-1) :: w


	do j= 0,ny-1
		do i = 0,nx-1
			w(i,j) = (w(i,j)-mean)/sdev
		end do
	end do


	return
end

!--------------------------------------------------------------------------
!Calculates laplacian for given 2D matrix input from python
!Validated
!--------------------------------------------------------------------------
subroutine laplacian_calculator(omega,laplacian,nx,ny,dx,dy)

!$ use omp_lib
implicit none
integer, intent(in) :: nx,ny
integer :: i,j
double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega
double precision,dimension(0:nx-1,0:ny-1),intent(out) :: laplacian
double precision,intent(in) :: dx, dy
double precision, dimension(:,:), allocatable :: omega_new
double precision :: lap_val

!Normalize omega
allocate(omega_new(-1:nx,-1:ny))

do j = 0,ny-1
  do i = 0,nx-1
    omega_new(i,j) = (omega(i,j))
  end do
end do

!BC Update
call periodic_bc_update(nx,ny,omega_new)

!Calculating laplacian
do j = 0,ny-1
  do i = 0,nx-1
    lap_val = (omega_new(i+1,j)+omega_new(i-1,j)-2.0d0*omega_new(i,j))/(dx*dx)
    laplacian(i,j) = lap_val + (omega_new(i,j+1)+omega_new(i,j-1)-2.0d0*omega_new(i,j))/(dy*dy)
  end do
end do

deallocate(omega_new)

return
end


!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sampling a field for ml based filter
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine conv_field_sampler(w,sampling_matrix,nx,ny)

	implicit none
	integer, intent(in) :: nx,ny
	integer :: i,j, sample
	double precision,dimension(0:nx-1,0:ny-1),intent(in) :: w
	double precision,dimension(0:nx*ny,0:8),intent(out) :: sampling_matrix
	double precision, dimension(:,:), allocatable :: w_new

	allocate(w_new(-1:nx,-1:ny))

	do j = 0,ny-1
	  do i = 0,nx-1
	    w_new(i,j) = (w(i,j))
	  end do
	end do

	!BC Update
	call periodic_bc_update(nx,ny,w_new)

	sample = 0
	do j = 0,ny-1
  		do i = 0,nx-1
		    sampling_matrix(sample,0) = w_new(i,j)
		    sampling_matrix(sample,1) = w_new(i,j+1)
		    sampling_matrix(sample,2) = w_new(i,j-1)
		    sampling_matrix(sample,3) = w_new(i+1,j)
		    sampling_matrix(sample,4) = w_new(i+1,j+1)
		    sampling_matrix(sample,5) = w_new(i+1,j-1)
		    sampling_matrix(sample,6) = w_new(i-1,j)
		    sampling_matrix(sample,7) = w_new(i-1,j+1)
		    sampling_matrix(sample,8) = w_new(i-1,j-1)

    		sample = sample + 1

  		end do
	end do

	deallocate(w_new)

	return
end

!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation with average shifting for backscatter
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator(jcf,jcad,laplacian,sfs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sfs

	integer :: i,j

	do j = 0,ny-1
		do i = 0,nx-1
			sfs(i,j) = jcf(i,j)-jcad(i,j)

			if (laplacian(i,j)<0 .and. sfs(i,j)>0) then
				sfs(i,j) = 0.0d0
			else if (laplacian(i,j)>0 .and. sfs(i,j)<0) then
				sfs(i,j) = 0.0d0
			end if

		end do
	end do

	return
end

!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation with average shifting for backscatter
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_weiss(w,s,jcf,jcad,laplacian,sfs,nx,ny,dx,dy)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,w,s,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sfs
	double precision, intent(in) :: dx, dy
	double precision, allocatable,dimension(:,:) :: u,v,stemp,weiss,nu_e
	double precision, allocatable,dimension(:,:) :: dudx,dudy,dvdx,dvdy
	double precision :: sigma

	integer :: i,j

	!Calculate SFS without truncation
	do j = 0,ny-1
		do i = 0,nx-1
			sfs(i,j) = jcf(i,j)-jcad(i,j)
		end do
	end do

	!Calculate velocities from streamfunction
	!allocate dummy array for periodic_bc_update
	allocate(stemp(-1:nx,-1:ny))
	do j = 0,ny-1
		do i = 0,nx-1
			stemp(i,j) = s(i,j)
		end do
	end do

	call periodic_bc_update(nx,ny,stemp)

	allocate(u(-1:nx,-1:ny))
	allocate(v(-1:nx,-1:ny))

	do j = 0,ny-1
		do i = 0,nx-1
			u(i,j) = (stemp(i,j+1)-stemp(i,j-1))/(2.0d0*dy)
			v(i,j) = -(stemp(i+1,j+1)-stemp(i-1,j))/(2.0d0*dx)
		end do
	end do

	deallocate(stemp)

	!Update velocity bcs
	call periodic_bc_update(nx,ny,u)
	call periodic_bc_update(nx,ny,v)

	!Calculate gradients of velocity
	allocate(dudx(0:nx-1,0:ny-1))
	allocate(dudy(0:nx-1,0:ny-1))
	allocate(dvdx(0:nx-1,0:ny-1))
	allocate(dvdy(0:nx-1,0:ny-1))

	do j = 0,ny-1
		do i = 0,nx-1
			dudx(i,j) = (u(i+1,j)-u(i-1,j))/(2.0d0*dx)
			dvdx(i,j) = (v(i+1,j)-v(i-1,j))/(2.0d0*dx)
			dudy(i,j) = (u(i,j+1)-u(i,j-1))/(2.0d0*dy)
			dvdy(i,j) = (v(i,j+1)-v(i,j-1))/(2.0d0*dy)
		end do
	end do

	deallocate(u,v)

	!calculate weiss criterion
	allocate(weiss(0:nx-1,0:ny-1))

	do j = 0,ny-1
		do i = 0,nx-1
			sigma = 0.5d0*((2.0d0*dudx(i,j))**2 + (2.0d0*dvdy(i,j))**2&
					+ 2.0d0*(dudy(i,j)+dvdx(i,j))**2) 
			weiss(i,j) = sigma - w(i,j)**2
		end do
	end do

	!Calculate eddy viscosities
	allocate(nu_e(0:nx-1,0:ny-1))
	do j = 0,ny-1
		do i = 0,nx-1
			nu_e(i,j) = sfs(i,j)/laplacian(i,j)
		end do
	end do

	!do truncation based on weiss criterion
	do j = 0,ny-1
		do i = 0,nx-1
			if (weiss(i,j)>7.0d0*w(i,j)**2.or.nu_e(i,j)<0.0d0) then
				sfs(i,j) = 0.0d0
			end if

			! if (nu_e(i,j)<0.0d0) then
			! 	sfs(i,j) = 0.0d0
			! end if
		end do
	end do

	deallocate(dudx,dudy,dvdx,dvdy,weiss)

	return
end



!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation 
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_bs_1(jcf,jcad,laplacian,sgs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sgs
	double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
	double precision, dimension(1:9) :: nu, lap
	double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av
	integer :: i,j,k


	tiny_val = 1.0d-10

	allocate(sgs_temp(-1:nx,-1:ny))
	allocate(lap_temp(-1:nx,-1:ny))

	!Reading return matrix
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1
		sgs_temp(i,j) = jcf(i,j)-jcad(i,j)
		lap_temp(i,j) = laplacian(i,j)
	  end do
	end do
	!$OMP END PARALLEL DO

	!Boundary conditions update
	!BC Update
	call periodic_bc_update(nx,ny,sgs_temp)
	call periodic_bc_update(nx,ny,lap_temp)


	!Local averaging
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1

		!Calculate eddy viscosities
		sgs_val = sgs_temp(i,j)
		lap_val = lap_temp(i,j)
		nu(1) = sgs_val/lap_val
		lap(1) = lap_val

		sgs_val = sgs_temp(i,j+1)
		lap_val = lap_temp(i,j+1)
		nu(2) = sgs_val/lap_val
		lap(2) = lap_val

		sgs_val = sgs_temp(i,j-1)
		lap_val = lap_temp(i,j-1)
		nu(3) = sgs_val/lap_val
		lap(3) = lap_val

		sgs_val = sgs_temp(i+1,j)
		lap_val = lap_temp(i+1,j)
		nu(4) = sgs_val/lap_val
		lap(4) = lap_val

		sgs_val = sgs_temp(i+1,j+1)
		lap_val = lap_temp(i+1,j+1)
		nu(5) = sgs_val/lap_val
		lap(5) = lap_val

		sgs_val = sgs_temp(i+1,j-1)
		lap_val = lap_temp(i+1,j-1)
		nu(6) = sgs_val/lap_val
		lap(6) = lap_val

		sgs_val = sgs_temp(i-1,j)
		lap_val = lap_temp(i-1,j)
		nu(7) = sgs_val/lap_val
		lap(7) = lap_val

		sgs_val = sgs_temp(i-1,j+1)
		lap_val = lap_temp(i-1,j+1)
		nu(8) = sgs_val/lap_val
		lap(8) = lap_val

		sgs_val = sgs_temp(i-1,j-1)
		lap_val = lap_temp(i-1,j-1)
		nu(9) = sgs_val/lap_val
		lap(9) = lap_val

		nu_av = 0.0d0
		lap_av = 0.0d0
		do k = 1,9
		    nu_av = nu_av + nu(k)
		    lap_av = lap_av + lap(k)
		end do
		nu_av = nu_av/9.0d0
		lap_av = lap_av/9.0d0


		if (nu(1)>tiny_val.and.nu_av>nu(1)) then
		   sgs(i,j) = nu(1)*laplacian(i,j)
		else
		   sgs(i,j) = 0.0d0
		end if

	 
	  end do
	end do

	deallocate(sgs_temp,lap_temp)

return
end


!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation - equation 27 (San IJCFD 2014)
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_bs_2(jcf,jcad,laplacian,sgs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sgs
	double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
	double precision, dimension(1:9) :: nu, lap
	double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av
	integer :: i,j,k


	tiny_val = 1.0d-10

	allocate(sgs_temp(-1:nx,-1:ny))
	allocate(lap_temp(-1:nx,-1:ny))

	!Reading return matrix
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1
		sgs_temp(i,j) = jcf(i,j)-jcad(i,j)
		lap_temp(i,j) = laplacian(i,j)
	  end do
	end do
	!$OMP END PARALLEL DO

	!Boundary conditions update
	!BC Update
	call periodic_bc_update(nx,ny,sgs_temp)
	call periodic_bc_update(nx,ny,lap_temp)


	!Local averaging
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1

		!Calculate eddy viscosities
		sgs_val = sgs_temp(i,j)
		lap_val = lap_temp(i,j)
		nu(1) = 4.0d0*sgs_val/lap_val
		lap(1) = lap_val

		sgs_val = sgs_temp(i,j+1)
		lap_val = lap_temp(i,j+1)
		nu(2) = 2.0d0*sgs_val/lap_val
		lap(2) = lap_val

		sgs_val = sgs_temp(i,j-1)
		lap_val = lap_temp(i,j-1)
		nu(3) = 2.0d0*sgs_val/lap_val
		lap(3) = lap_val

		sgs_val = sgs_temp(i+1,j)
		lap_val = lap_temp(i+1,j)
		nu(4) = 2.0d0*sgs_val/lap_val
		lap(4) = lap_val

		sgs_val = sgs_temp(i+1,j+1)
		lap_val = lap_temp(i+1,j+1)
		nu(5) = sgs_val/lap_val
		lap(5) = lap_val

		sgs_val = sgs_temp(i+1,j-1)
		lap_val = lap_temp(i+1,j-1)
		nu(6) = sgs_val/lap_val
		lap(6) = lap_val

		sgs_val = sgs_temp(i-1,j)
		lap_val = lap_temp(i-1,j)
		nu(7) = 2.0d0*sgs_val/lap_val
		lap(7) = lap_val

		sgs_val = sgs_temp(i-1,j+1)
		lap_val = lap_temp(i-1,j+1)
		nu(8) = sgs_val/lap_val
		lap(8) = lap_val

		sgs_val = sgs_temp(i-1,j-1)
		lap_val = lap_temp(i-1,j-1)
		nu(9) = sgs_val/lap_val
		lap(9) = lap_val

		nu_av = 0.0d0
		lap_av = 0.0d0
		do k = 1,9
		    nu_av = nu_av + nu(k)
		    lap_av = lap_av + lap(k)
		end do
		nu_av = nu_av/16.0d0
		lap_av = lap_av/16.0d0


		if (nu(1)>tiny_val.and.nu_av>nu(1)) then
		   sgs(i,j) = nu(1)/4.0d0*laplacian(i,j)
		else
		   sgs(i,j) = 0.0d0
		end if

	 
	  end do
	end do

	deallocate(sgs_temp,lap_temp)

return
end


!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation - Equation 31
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_bs_3(jcf,jcad,laplacian,sgs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sgs
	double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
	double precision, dimension(1:9) :: nu, lap
	double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av
	integer :: i,j,k


	tiny_val = 1.0d-10

	allocate(sgs_temp(-1:nx,-1:ny))
	allocate(lap_temp(-1:nx,-1:ny))

	!Reading return matrix
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1
		sgs_temp(i,j) = jcf(i,j)-jcad(i,j)
		lap_temp(i,j) = laplacian(i,j)
	  end do
	end do
	!$OMP END PARALLEL DO

	!Boundary conditions update
	!BC Update
	call periodic_bc_update(nx,ny,sgs_temp)
	call periodic_bc_update(nx,ny,lap_temp)


	!Local averaging
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1

		!Calculate eddy viscosities
		sgs_val = sgs_temp(i,j)
		lap_val = lap_temp(i,j)
		nu(1) = 8.0d0*sgs_val/lap_val
		lap(1) = lap_val

		sgs_val = sgs_temp(i,j+1)
		lap_val = lap_temp(i,j+1)
		nu(2) = sgs_val/lap_val
		lap(2) = lap_val

		sgs_val = sgs_temp(i,j-1)
		lap_val = lap_temp(i,j-1)
		nu(3) = sgs_val/lap_val
		lap(3) = lap_val

		sgs_val = sgs_temp(i+1,j)
		lap_val = lap_temp(i+1,j)
		nu(4) = sgs_val/lap_val
		lap(4) = lap_val

		sgs_val = sgs_temp(i+1,j+1)
		lap_val = lap_temp(i+1,j+1)
		nu(5) = sgs_val/lap_val
		lap(5) = lap_val

		sgs_val = sgs_temp(i+1,j-1)
		lap_val = lap_temp(i+1,j-1)
		nu(6) = sgs_val/lap_val
		lap(6) = lap_val

		sgs_val = sgs_temp(i-1,j)
		lap_val = lap_temp(i-1,j)
		nu(7) = sgs_val/lap_val
		lap(7) = lap_val

		sgs_val = sgs_temp(i-1,j+1)
		lap_val = lap_temp(i-1,j+1)
		nu(8) = sgs_val/lap_val
		lap(8) = lap_val

		sgs_val = sgs_temp(i-1,j-1)
		lap_val = lap_temp(i-1,j-1)
		nu(9) = sgs_val/lap_val
		lap(9) = lap_val

		nu_av = 0.0d0
		lap_av = 0.0d0
		do k = 1,9
		    nu_av = nu_av + nu(k)
		    lap_av = lap_av + lap(k)
		end do
		nu_av = nu_av/16.0d0
		lap_av = lap_av/16.0d0


		if (nu(1)>tiny_val.and.nu_av>nu(1)) then
		   sgs(i,j) = nu(1)/8.0d0*laplacian(i,j)
		else
		   sgs(i,j) = 0.0d0
		end if

	 
	  end do
	end do

	deallocate(sgs_temp,lap_temp)

return
end


!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation 
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_bs_4(jcf,jcad,laplacian,sgs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sgs
	double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
	double precision, dimension(1:9) :: nu, lap
	double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av
	integer :: i,j,k


	tiny_val = 1.0d-10

	allocate(sgs_temp(-1:nx,-1:ny))
	allocate(lap_temp(-1:nx,-1:ny))

	!Reading return matrix
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1
		sgs_temp(i,j) = jcf(i,j)-jcad(i,j)
		lap_temp(i,j) = laplacian(i,j)
	  end do
	end do
	!$OMP END PARALLEL DO

	!Boundary conditions update
	!BC Update
	call periodic_bc_update(nx,ny,sgs_temp)
	call periodic_bc_update(nx,ny,lap_temp)


	!Local averaging
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1

		!Calculate eddy viscosities
		sgs_val = sgs_temp(i,j)
		lap_val = lap_temp(i,j)
		nu(1) = 4.0d0*sgs_val/lap_val
		lap(1) = lap_val

		sgs_val = sgs_temp(i,j+1)
		lap_val = lap_temp(i,j+1)
		nu(2) = sgs_val/lap_val
		lap(2) = lap_val

		sgs_val = sgs_temp(i,j-1)
		lap_val = lap_temp(i,j-1)
		nu(3) = sgs_val/lap_val
		lap(3) = lap_val

		sgs_val = sgs_temp(i+1,j)
		lap_val = lap_temp(i+1,j)
		nu(4) = sgs_val/lap_val
		lap(4) = lap_val

		sgs_val = sgs_temp(i+1,j+1)
		lap_val = lap_temp(i+1,j+1)
		nu(5) = 0.0d0*sgs_val/lap_val
		lap(5) = lap_val

		sgs_val = sgs_temp(i+1,j-1)
		lap_val = lap_temp(i+1,j-1)
		nu(6) = 0.0d0*sgs_val/lap_val
		lap(6) = lap_val

		sgs_val = sgs_temp(i-1,j)
		lap_val = lap_temp(i-1,j)
		nu(7) = sgs_val/lap_val
		lap(7) = lap_val

		sgs_val = sgs_temp(i-1,j+1)
		lap_val = lap_temp(i-1,j+1)
		nu(8) = 0.0d0*sgs_val/lap_val
		lap(8) = lap_val

		sgs_val = sgs_temp(i-1,j-1)
		lap_val = lap_temp(i-1,j-1)
		nu(9) = 0.0d0*sgs_val/lap_val
		lap(9) = lap_val

		nu_av = 0.0d0
		lap_av = 0.0d0
		do k = 1,9
		    nu_av = nu_av + nu(k)
		    lap_av = lap_av + lap(k)
		end do
		nu_av = nu_av/8.0d0
		lap_av = lap_av/8.0d0


		if (nu(1)>tiny_val.and.nu_av>nu(1)) then
		   sgs(i,j) = nu(1)/4.0d0*laplacian(i,j)
		else
		   sgs(i,j) = 0.0d0
		end if

	 
	  end do
	end do

	deallocate(sgs_temp,lap_temp)

return
end


!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation - Equation 31
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_bs_5(jcf,jcad,laplacian,sgs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sgs
	double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
	double precision, dimension(1:9) :: nu, lap
	double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av
	integer :: i,j,k


	tiny_val = 1.0d-10

	allocate(sgs_temp(-1:nx,-1:ny))
	allocate(lap_temp(-1:nx,-1:ny))

	!Reading return matrix
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1
		sgs_temp(i,j) = jcf(i,j)-jcad(i,j)
		lap_temp(i,j) = laplacian(i,j)
	  end do
	end do
	!$OMP END PARALLEL DO

	!Boundary conditions update
	!BC Update
	call periodic_bc_update(nx,ny,sgs_temp)
	call periodic_bc_update(nx,ny,lap_temp)


	!Local averaging
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1

		!Calculate eddy viscosities
		sgs_val = sgs_temp(i,j)
		lap_val = lap_temp(i,j)
		nu(1) = 6.0d0*sgs_val/lap_val
		lap(1) = lap_val

		sgs_val = sgs_temp(i,j+1)
		lap_val = lap_temp(i,j+1)
		nu(2) = sgs_val/lap_val
		lap(2) = lap_val

		sgs_val = sgs_temp(i,j-1)
		lap_val = lap_temp(i,j-1)
		nu(3) = sgs_val/lap_val
		lap(3) = lap_val

		sgs_val = sgs_temp(i+1,j)
		lap_val = lap_temp(i+1,j)
		nu(4) = sgs_val/lap_val
		lap(4) = lap_val

		sgs_val = sgs_temp(i+1,j+1)
		lap_val = lap_temp(i+1,j+1)
		nu(5) = 0.5d0*sgs_val/lap_val
		lap(5) = lap_val

		sgs_val = sgs_temp(i+1,j-1)
		lap_val = lap_temp(i+1,j-1)
		nu(6) = 0.5d0*sgs_val/lap_val
		lap(6) = lap_val

		sgs_val = sgs_temp(i-1,j)
		lap_val = lap_temp(i-1,j)
		nu(7) = sgs_val/lap_val
		lap(7) = lap_val

		sgs_val = sgs_temp(i-1,j+1)
		lap_val = lap_temp(i-1,j+1)
		nu(8) = 0.5d0*sgs_val/lap_val
		lap(8) = lap_val

		sgs_val = sgs_temp(i-1,j-1)
		lap_val = lap_temp(i-1,j-1)
		nu(9) = 0.5d0*sgs_val/lap_val
		lap(9) = lap_val

		nu_av = 0.0d0
		lap_av = 0.0d0
		do k = 1,9
		    nu_av = nu_av + nu(k)
		    lap_av = lap_av + lap(k)
		end do
		nu_av = nu_av/12.0d0
		lap_av = lap_av/12.0d0


		if (nu(1)>tiny_val.and.nu_av>nu(1)) then
		   sgs(i,j) = nu(1)/6.0d0*laplacian(i,j)
		else
		   sgs(i,j) = 0.0d0
		end if

	 
	  end do
	end do

	deallocate(sgs_temp,lap_temp)

return
end

!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
!subroutine for sfs calculation - Equation 31
!----------------------------------------------------------------------------------------------------!
!----------------------------------------------------------------------------------------------------!
subroutine ml_sfs_calculator_bs_6(jcf,jcad,laplacian,sgs,nx,ny)
	implicit none

	integer, intent(in) :: nx, ny
	double precision, intent(in), dimension(0:nx-1,0:ny-1) :: jcf,jcad,laplacian
	double precision, intent(out), dimension(0:nx-1,0:ny-1) :: sgs
	double precision, dimension(:,:), allocatable :: sgs_temp, lap_temp
	double precision, dimension(1:9) :: nu, lap
	double precision :: sgs_val, lap_val, nu_av, tiny_val, lap_av
	integer :: i,j,k


	tiny_val = 1.0d-10

	allocate(sgs_temp(-1:nx,-1:ny))
	allocate(lap_temp(-1:nx,-1:ny))

	!Reading return matrix
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1
		sgs_temp(i,j) = jcf(i,j)-jcad(i,j)
		lap_temp(i,j) = laplacian(i,j)
	  end do
	end do
	!$OMP END PARALLEL DO

	!Boundary conditions update
	!BC Update
	call periodic_bc_update(nx,ny,sgs_temp)
	call periodic_bc_update(nx,ny,lap_temp)


	!Local averaging
	!$OMP PARALLEL DO
	do j = 0,ny-1
	  do i = 0,nx-1

		!Calculate eddy viscosities
		sgs_val = sgs_temp(i,j)
		lap_val = lap_temp(i,j)
		nu(1) = 8.0d0*sgs_val/lap_val
		lap(1) = lap_val

		sgs_val = sgs_temp(i,j+1)
		lap_val = lap_temp(i,j+1)
		nu(2) = 2.0d0*sgs_val/lap_val
		lap(2) = lap_val

		sgs_val = sgs_temp(i,j-1)
		lap_val = lap_temp(i,j-1)
		nu(3) = 2.0d0*sgs_val/lap_val
		lap(3) = lap_val

		sgs_val = sgs_temp(i+1,j)
		lap_val = lap_temp(i+1,j)
		nu(4) = 2.0d0*sgs_val/lap_val
		lap(4) = lap_val

		sgs_val = sgs_temp(i+1,j+1)
		lap_val = lap_temp(i+1,j+1)
		nu(5) = 0.5d0*sgs_val/lap_val
		lap(5) = lap_val

		sgs_val = sgs_temp(i+1,j-1)
		lap_val = lap_temp(i+1,j-1)
		nu(6) = 0.5d0*sgs_val/lap_val
		lap(6) = lap_val

		sgs_val = sgs_temp(i-1,j)
		lap_val = lap_temp(i-1,j)
		nu(7) = 2.0d0*sgs_val/lap_val
		lap(7) = lap_val

		sgs_val = sgs_temp(i-1,j+1)
		lap_val = lap_temp(i-1,j+1)
		nu(8) = 0.5d0*sgs_val/lap_val
		lap(8) = lap_val

		sgs_val = sgs_temp(i-1,j-1)
		lap_val = lap_temp(i-1,j-1)
		nu(9) = 0.5d0*sgs_val/lap_val
		lap(9) = lap_val

		nu_av = 0.0d0
		lap_av = 0.0d0
		do k = 1,9
		    nu_av = nu_av + nu(k)
		    lap_av = lap_av + lap(k)
		end do
		nu_av = nu_av/18.0d0
		lap_av = lap_av/18.0d0


		if (nu(1)>tiny_val.and.nu_av>nu(1)) then
		   sgs(i,j) = nu(1)/8.0d0*laplacian(i,j)
		else
		   sgs(i,j) = 0.0d0
		end if

	 
	  end do
	end do

	deallocate(sgs_temp,lap_temp)

return
end





!--------------------------------------------------------------------------
!Subroutine for reshaping Keras prediction into nx x ny shape
!The return matrix here contains values for filtered field
!Validated
!--------------------------------------------------------------------------

subroutine field_reshape(return_matrix,nrows,ncols,field,nx,ny)

!$ use omp_lib
implicit none

integer, intent(in) :: nx,ny,nrows,ncols
integer :: i,j, sample
double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
double precision, dimension(0:nx-1,0:ny-1),intent(inout) :: field

sample = 0
!Reading return matrix
!$OMP PARALLEL DO
do j = 0,ny-1
  do i = 0,nx-1
    field(i,j) = return_matrix(sample,1)
    sample = sample + 1
  end do
end do
!$OMP END PARALLEL DO


return
end
