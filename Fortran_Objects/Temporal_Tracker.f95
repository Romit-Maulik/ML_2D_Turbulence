!--------------------------------------------------------------------------
!Track the classification of the machine learning framework
!--------------------------------------------------------------------------
subroutine classification_tracker(return_matrix,nrows,ncols,class_tracker,n_classes)
	implicit none
	integer, intent(in) :: nrows, ncols, n_classes
	double precision, dimension(0:nrows,ncols),intent(in) :: return_matrix
	integer*8, dimension(n_classes),intent(inout) :: class_tracker
	integer :: i

	class_tracker(1) = 0
	class_tracker(2) = 0
	class_tracker(3) = 0


	! Doing sampling of the entire domain for tracker
	do i = 0, nrows
		if (return_matrix(i,1)>return_matrix(i,2).and.return_matrix(i,1)>return_matrix(i,3)) then
        	class_tracker(1) = class_tracker(1)+ 1      !AD prediction
		else if (return_matrix(i,2)>return_matrix(i,1).and.return_matrix(i,2)>return_matrix(i,3)) then
		    class_tracker(2) = class_tracker(2)+ 1      !SM prediction
		else
		    class_tracker(3) = class_tracker(3)+ 1      !NM prediction
		end if
	end do

	return
end subroutine


!--------------------------------------------------------------------------
!Track the turbulence kinetic energy in the field
!--------------------------------------------------------------------------
subroutine tke_tracker(tke,psi,nx,ny,dx,dy)
	implicit none
	integer, intent(in) :: nx, ny
	double precision,dimension(0:nx-1,0:ny-1),intent(in) :: psi
	double precision, intent(in) :: dx, dy
	double precision, intent(out) :: tke
	double precision :: u_m, v_m
	double precision, dimension(:,:), allocatable :: psi_new,u,v
	integer :: i, j

	allocate(psi_new(-1:nx,-1:ny))
	allocate(u(0:nx-1,0:ny-1),v(0:nx-1,0:ny-1))

	do j = 0,ny-1
	  do i = 0,nx-1
	    psi_new(i,j) = psi(i,j)
	  end do
	end do

	!BC Update
	call periodic_bc_update(nx,ny,psi_new)

	!Calculate velocity components
	u_m = 0.0d0
	v_m = 0.0d0
	do j = 0,ny-1
		do i = 0, nx-1
			u(i,j) = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0d0*dy)
			v(i,j) = -(psi_new(i+1,j)-psi_new(i-1,j))/(2.0d0*dx)

			u_m = u_m + u(i,j)
			v_m = v_m + v(i,j)
		end do
	end do

	! Calculate means
	u_m = u_m/dfloat(nx*ny)
	v_m = v_m/dfloat(nx*ny)

	tke = 0.0d0
	! Calculate fluctuations and tke
	do j = 0,ny-1
		do i = 0, nx-1
			tke = tke + 0.5d0*((u_m - u(i,j))*(u_m - u(i,j)) + (v_m - v(i,j))*(v_m - v(i,j)))
		end do
	end do

	tke = tke/dfloat(nx*ny)

	deallocate(psi_new,u,v)

	return
end subroutine



!--------------------------------------------------------------------------
!Track the enstrophy in the field
!--------------------------------------------------------------------------
subroutine enstrophy_tracker(ens,omega,nx,ny)
	implicit none
	integer, intent(in) :: nx, ny
	double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega
	double precision, intent(out) :: ens
	integer :: i, j

	!Calculate velocity components
	ens = 0.0d0
	do j = 0,ny-1
		do i = 0, nx-1
			ens = ens + omega(i,j)**2
		end do
	end do

	! Calculate means
	ens = ens/dfloat(nx*ny)

	return
end subroutine


!--------------------------------------------------------------------------
!Track the vorticty variance in the field
!--------------------------------------------------------------------------
subroutine vort_var_tracker(var,omega,nx,ny)
	implicit none
	integer, intent(in) :: nx, ny
	double precision,dimension(0:nx-1,0:ny-1),intent(in) :: omega
	double precision, intent(out) :: var
	integer :: i, j
	double precision :: om

	! Calculate mean vorticity
	om = 0.0d0
	do j = 0,ny-1
		do i = 0, nx-1
			om = om + omega(i,j)
		end do
	end do
	om = om/dfloat(nx*ny)

	!Calculate velocity components
	var = 0.0d0
	do j = 0,ny-1
		do i = 0, nx-1
			var = var + (omega(i,j)-om)**2
		end do
	end do
	var = var/dfloat(nx*ny)

	return
end subroutine


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