!------------------------------------------------------------------
!Calculates right-hand-side for TVD RK3 implementations
!Second-order Arakawa Jacobian calculation
!Validated
!------------------------------------------------------------------
subroutine rhs_periodic(psi,omega,f,nx,ny,dx,dy,Re_N)
!$ use omp_lib
implicit none


integer, intent(in) :: nx, ny
integer :: i, j
double precision, intent(in) :: dx, dy, Re_N
double precision, dimension(0:nx-1,0:ny-1),intent(in)::psi,omega
double precision, dimension(0:nx-1,0:ny-1),intent(out)::f
double precision :: jj1, jj2, jj3, d2wdy2, d2wdx2

double precision, dimension(:,:), allocatable :: psi_new, omega_new

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do
!$OMP END PARALLEL DO

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

d2wdy2 = (omega_new(i, j+1) + omega_new(i, j-1) - 2.0 * omega_new(i, j)) / (dy * dy)
d2wdx2 = (omega_new(i+1, j) + omega_new(i-1, j) - 2.0 * omega_new(i, j)) / (dx * dx)

f(i, j) = (-(jj1 + jj2 + jj3)/3.0 + 1.0 / Re_n * (d2wdy2 + d2wdx2))

end do
end do
!$OMP END PARALLEL DO


deallocate(psi_new,omega_new)
end subroutine



!------------------------------------------------------------------
!Calculates right-hand-side for TVD RK3 implementations
!Uses FOU for calculation of nonlinear term - ILES
!------------------------------------------------------------------
subroutine rhs_periodic_fou(psi,omega,f,nx,ny,dx,dy,Re_N)
!$ use omp_lib
implicit none


integer, intent(in) :: nx, ny
integer :: i, j
double precision, intent(in) :: dx, dy, Re_N
double precision, dimension(0:nx-1,0:ny-1),intent(in)::psi,omega
double precision, dimension(0:nx-1,0:ny-1),intent(out)::f
double precision :: d2wdy2, d2wdx2, j1, j2

double precision, dimension(:,:), allocatable :: psi_new, omega_new
double precision, dimension(:,:), allocatable :: u,v
double precision, dimension(:,:), allocatable :: up,um,vp,vm

allocate(psi_new(-1:nx,-1:ny))
allocate(omega_new(-1:nx,-1:ny))

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do
!$OMP END PARALLEL DO

call periodic_bc_update(nx,ny,psi_new)
call periodic_bc_update(nx,ny,omega_new)

allocate(u(-1:nx,-1:ny))
allocate(v(-1:nx,-1:ny))

do j = 0,ny-1
do i = 0,nx-1
u(i,j) = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0*dy)
v(i,j) = -(psi_new(i+1,j)-psi_new(i-1,j))/(2.0*dx)
end do
end do

call periodic_bc_update(nx,ny,u)
call periodic_bc_update(nx,ny,v)


allocate(up(0:nx-1,0:ny-1))
allocate(um(0:nx-1,0:ny-1))
allocate(vp(0:nx-1,0:ny-1))
allocate(vm(0:nx-1,0:ny-1))


do j = 0,ny-1
	do i = 0,nx-1
		up(i,j) = 0.5d0*(u(i+1,j)+u(i,j))
		um(i,j) = 0.5d0*(u(i-1,j)+u(i,j))
		vp(i,j) = 0.5d0*(v(i,j+1)+v(i,j))
		vm(i,j) = 0.5d0*(v(i,j-1)+v(i,j))
	end do
end do

do j = 0,ny-1
	do i = 0,nx-1
		j1 = (up(i,j)-dabs(up(i,j)))*omega_new(i+1,j)
		j1 = j1 + (up(i,j)+dabs(up(i,j))-um(i,j)+dabs(um(i,j)))*omega_new(i,j)
		j1 = j1 - (um(i,j)+dabs(um(i,j)))*omega_new(i-1,j)
		j1 = j1/dx

		j2 = (vp(i,j)-dabs(vp(i,j)))*omega_new(i,j+1)
		j2 = j2 + (vp(i,j)+dabs(vp(i,j))-vm(i,j)+dabs(vm(i,j)))*omega_new(i,j)
		j2 = j2 - (vm(i,j)+dabs(vm(i,j)))*omega_new(i,j-1)
		j2 = j2/dy

		d2wdy2 = (omega_new(i, j+1) + omega_new(i, j-1) - 2.0 * omega_new(i, j)) / (dy * dy)
		d2wdx2 = (omega_new(i+1, j) + omega_new(i-1, j) - 2.0 * omega_new(i, j)) / (dx * dx)

		f(i, j) = (-(j1+j2) + 1.0 / Re_n * (d2wdy2 + d2wdx2))
	end do
end do


deallocate(psi_new,omega_new,u,v)
deallocate(up,um,vp,vm)


end subroutine



!------------------------------------------------------------------
!Calculates right-hand-side for TVD RK3 implementations
!Second/Third-order accurate upwind
! Salih, IIST, Streamfunction-Vorticity formulation
!------------------------------------------------------------------
subroutine rhs_periodic_iles(psi,omega,f,nx,ny,dx,dy,Re_N)
!$ use omp_lib
implicit none


integer, intent(in) :: nx, ny
integer :: i, j
double precision, intent(in) :: dx, dy, Re_N
double precision, dimension(0:nx-1,0:ny-1),intent(in)::psi,omega
double precision, dimension(0:nx-1,0:ny-1),intent(out)::f
double precision :: d2wdy2, d2wdx2, j1, qq

double precision, dimension(:,:), allocatable :: psi_new, omega_new
double precision, dimension(:,:), allocatable :: u,v
double precision, dimension(:,:), allocatable :: up,um,vp,vm
double precision, dimension(:,:), allocatable :: wxp,wxm,wyp,wym


qq = 0.5d0

allocate(psi_new(-2:nx+1,-2:ny+1))
allocate(omega_new(-2:nx+1,-2:ny+1))

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do
!$OMP END PARALLEL DO

call periodic_bc_update_quick(nx,ny,psi_new)
call periodic_bc_update_quick(nx,ny,omega_new)

allocate(u(0:nx-1,0:ny-1))
allocate(v(0:nx-1,0:ny-1))
allocate(um(0:nx-1,0:ny-1))
allocate(up(0:nx-1,0:ny-1))
allocate(vm(0:nx-1,0:ny-1))
allocate(vp(0:nx-1,0:ny-1))
allocate(wxp(0:nx-1,0:ny-1))
allocate(wxm(0:nx-1,0:ny-1))
allocate(wyp(0:nx-1,0:ny-1))
allocate(wym(0:nx-1,0:ny-1))


do j = 0,ny-1
do i = 0,nx-1

u(i,j) = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0*dy)
v(i,j) = -(psi_new(i+1,j)-psi_new(i-1,j))/(2.0*dx)

up(i,j) = max(u(i,j),0.0d0)
vp(i,j) = max(v(i,j),0.0d0)
um(i,j) = min(u(i,j),0.0d0)
vm(i,j) = min(v(i,j),0.0d0)

wxm(i,j) = (omega_new(i-2,j) - 3.0d0*omega_new(i-1,j) + 3.0d0*omega_new(i,j) - omega_new(i+1,j))/(3.0d0*dx)
wxp(i,j) = (omega_new(i-1,j) - 3.0d0*omega_new(i,j) + 3.0d0*omega_new(i+1,j) - omega_new(i+2,j))/(3.0d0*dx)

wym(i,j) = (omega_new(i,j-2) - 3.0d0*omega_new(i,j-1) + 3.0d0*omega_new(i,j) - omega_new(i,j+1))/(3.0d0*dx)
wyp(i,j) = (omega_new(i,j-1) - 3.0d0*omega_new(i,j) + 3.0d0*omega_new(i,j+1) - omega_new(i,j+2))/(3.0d0*dx)

end do
end do


do j = 0,ny-1
	do i = 0,nx-1
		j1 = u(i,j)*(omega_new(i+1,j)-omega_new(i-1,j))/(2.0d0*dx)
		j1 = j1 + v(i,j)*(omega_new(i,j+1)-omega_new(i,j-1))/(2.0d0*dy)
		j1 = j1 + qq*(up(i,j)*wxm(i,j) + um(i,j)*wxp(i,j))
		j1 = j1 + qq*(vp(i,j)*wym(i,j) + vm(i,j)*wyp(i,j))

		d2wdy2 = (omega_new(i, j+1) + omega_new(i, j-1) - 2.0 * omega_new(i, j)) / (dy * dy)
		d2wdx2 = (omega_new(i+1, j) + omega_new(i-1, j) - 2.0 * omega_new(i, j)) / (dx * dx)

		f(i, j) = (-(j1) + 1.0 / Re_n * (d2wdy2 + d2wdx2))
	end do
end do


deallocate(psi_new,omega_new,u,v)
deallocate(up,um,vp,vm,wxp,wxm,wyp,wym)


end subroutine



!------------------------------------------------------------------
!Calculates right-hand-side for TVD RK3 implementations
! Switch between central and upwind according to model
! Salih, IIST, Streamfunction-Vorticity formulation
!------------------------------------------------------------------
subroutine rhs_periodic_iles_switching(psi,omega,f,nx,ny,dx,dy,Re_N,switching_matrix,s_rows,s_cols)
!$ use omp_lib
implicit none


integer, intent(in) :: nx, ny, s_rows, s_cols
integer :: i, j, sample
double precision, intent(in) :: dx, dy, Re_N
double precision, dimension(0:nx-1,0:ny-1),intent(in)::psi,omega
double precision, dimension(0:nx-1,0:ny-1),intent(out)::f
double precision, dimension(0:s_rows,s_cols),intent(in)::switching_matrix

double precision :: d2wdy2, d2wdx2, j1, qq, jj1, jj2, jj3

double precision, dimension(:,:), allocatable :: psi_new, omega_new
double precision, dimension(:,:), allocatable :: u,v, f_iles, f_cen
double precision, dimension(:,:), allocatable :: up,um,vp,vm
double precision, dimension(:,:), allocatable :: wxp,wxm,wyp,wym
integer, dimension(:,:),allocatable :: label_field


allocate(label_field(0:nx-1,0:ny-1))
allocate(f_iles(0:nx-1,0:ny-1))
allocate(f_cen(0:nx-1,0:ny-1))

!Reshape switching_matrix
sample = 0
do j = 0,ny-1
  do i = 0,nx-1

    if (switching_matrix(sample,1)>switching_matrix(sample,2).and.switching_matrix(sample,1)>switching_matrix(sample,3)) then
        label_field(i,j) = 0		!Classifying AD - central will be used
    else if (switching_matrix(sample,2)>switching_matrix(sample,1).and.switching_matrix(sample,2)>switching_matrix(sample,3)) then
        label_field(i,j) = 1       !Classifying Smag - Use upwinding because dissipation is required
    else
        label_field(i,j) = 0	   !Classifying no model - Use upwinding
    end if

    sample = sample + 1 
  end do
end do

qq = 0.5d0

allocate(psi_new(-2:nx+1,-2:ny+1))
allocate(omega_new(-2:nx+1,-2:ny+1))

!$OMP PARALLEL DO
do j = 0,ny-1
do i = 0,nx-1
psi_new(i,j) = psi(i,j)
omega_new(i,j) = omega(i,j)
end do
end do
!$OMP END PARALLEL DO

call periodic_bc_update_quick(nx,ny,psi_new)
call periodic_bc_update_quick(nx,ny,omega_new)

allocate(u(0:nx-1,0:ny-1))
allocate(v(0:nx-1,0:ny-1))
allocate(um(0:nx-1,0:ny-1))
allocate(up(0:nx-1,0:ny-1))
allocate(vm(0:nx-1,0:ny-1))
allocate(vp(0:nx-1,0:ny-1))
allocate(wxp(0:nx-1,0:ny-1))
allocate(wxm(0:nx-1,0:ny-1))
allocate(wyp(0:nx-1,0:ny-1))
allocate(wym(0:nx-1,0:ny-1))

!Third-order upwind estimate
do j = 0,ny-1
do i = 0,nx-1

u(i,j) = (psi_new(i,j+1)-psi_new(i,j-1))/(2.0*dy)
v(i,j) = -(psi_new(i+1,j)-psi_new(i-1,j))/(2.0*dx)

up(i,j) = max(u(i,j),0.0d0)
vp(i,j) = max(v(i,j),0.0d0)
um(i,j) = min(u(i,j),0.0d0)
vm(i,j) = min(v(i,j),0.0d0)

wxm(i,j) = (omega_new(i-2,j) - 3.0d0*omega_new(i-1,j) + 3.0d0*omega_new(i,j) - omega_new(i+1,j))/(3.0d0*dx)
wxp(i,j) = (omega_new(i-1,j) - 3.0d0*omega_new(i,j) + 3.0d0*omega_new(i+1,j) - omega_new(i+2,j))/(3.0d0*dx)

wym(i,j) = (omega_new(i,j-2) - 3.0d0*omega_new(i,j-1) + 3.0d0*omega_new(i,j) - omega_new(i,j+1))/(3.0d0*dx)
wyp(i,j) = (omega_new(i,j-1) - 3.0d0*omega_new(i,j) + 3.0d0*omega_new(i,j+1) - omega_new(i,j+2))/(3.0d0*dx)

end do
end do


do j = 0,ny-1
	do i = 0,nx-1
		j1 = u(i,j)*(omega_new(i+1,j)-omega_new(i-1,j))/(2.0d0*dx)
		j1 = j1 + v(i,j)*(omega_new(i,j+1)-omega_new(i,j-1))/(2.0d0*dy)
		j1 = j1 + qq*(up(i,j)*wxm(i,j) + um(i,j)*wxp(i,j))
		j1 = j1 + qq*(vp(i,j)*wym(i,j) + vm(i,j)*wyp(i,j))

		d2wdy2 = (omega_new(i, j+1) + omega_new(i, j-1) - 2.0 * omega_new(i, j)) / (dy * dy)
		d2wdx2 = (omega_new(i+1, j) + omega_new(i-1, j) - 2.0 * omega_new(i, j)) / (dx * dx)

		f_iles(i, j) = (-(j1) + 1.0 / Re_n * (d2wdy2 + d2wdx2))
	end do
end do




!Second order Arakawa estimate
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

d2wdy2 = (omega_new(i, j+1) + omega_new(i, j-1) - 2.0 * omega_new(i, j)) / (dy * dy)
d2wdx2 = (omega_new(i+1, j) + omega_new(i-1, j) - 2.0 * omega_new(i, j)) / (dx * dx)

f_cen(i, j) = (-(jj1 + jj2 + jj3)/3.0 + 1.0 / Re_n * (d2wdy2 + d2wdx2))

end do
end do


!Do switching on the fly
do j = 0,ny-1
	do i = 0,nx-1
		if (label_field(i,j)==0) then
			f(i,j) = f_cen(i,j)

		else
			f(i,j) = f_iles(i,j)
		end if
	end do
end do



deallocate(psi_new,omega_new,u,v)
deallocate(up,um,vp,vm,wxp,wxm,wyp,wym,label_field,f_iles,f_cen)


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


!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!Validated
!---------------------------------------------------------------------------!
subroutine periodic_bc_update_quick(nx,ny,u)
implicit none

integer :: nx, ny, i, j
double precision, dimension(-2:nx+1,-2:ny+1) :: u

do i = 0,nx-1
u(i,-1) = u(i,ny-1)
u(i,-2) = u(i,ny-2)
u(i,ny) = u(i,0)
u(i,ny+1) = u(i,1)
end do

do j = -2,ny+1
u(-2,j) = u(nx-2,j)
u(-1,j) = u(nx-1,j)
u(nx,j) = u(0,j)
u(nx+1,j) = u(1,j)
end do

end subroutine




!---------------------------------------------------------------------------!
!Compute 2D vorticity field from the energy spectrum
!Periodic, equidistant grid
!Validated
!---------------------------------------------------------------------------!
subroutine hit_init_cond(w_org,nx,ny,dx,dy)
implicit none
integer,intent(inout)::nx,ny
double precision,intent(inout) ::w_org(0:nx-1,0:ny-1)
double precision, intent(in) :: dx, dy
integer :: ni,nj,ii,jj
double precision,dimension(:,:),allocatable ::w
double precision ::ran,pi,kk,E4
double precision,parameter:: tiny=1.0d-10
double precision,allocatable ::data1d(:),phase2d(:,:,:),ksi(:,:),eta(:,:)
double precision,allocatable ::kx(:),ky(:),ww(:,:)
integer::i,j,k,isign,ndim,nn(2),seed

allocate(w(-2:nx+2,-2:ny+2))

w = 0.0d0

seed = 19

!expand it to dns grid

ni = nx
nj = ny

nx = 2048
ny = 2048

ii = nx/ni
jj = ny/nj

ndim =2
nn(1)=nx
nn(2)=ny

allocate(kx(0:nx-1),ky(0:ny-1))
allocate(ksi(0:nx/2,0:ny/2),eta(0:nx/2,0:ny/2))
allocate(data1d(2*nx*ny))
allocate(phase2d(2,0:nx-1,0:ny-1))
allocate(ww(0:nx,0:ny))

!Set seed for the random number generator between [0,1]
CALL RANDOM_SEED(seed)

pi = 4.0d0*datan(1.0d0)


!Wave numbers 
do i=0,nx/2-1
kx(i)      = dfloat(i)
kx(i+nx/2) = dfloat(i-nx/2)
end do
kx(0) = tiny

do j=0,ny/2-1
ky(j)      = dfloat(j)
ky(j+ny/2) = dfloat(j-ny/2)
end do
ky(0) = tiny

!Random numbers in the first quadrant
do j=0,ny/2
do i=0,nx/2
CALL RANDOM_NUMBER(ran)
ksi(i,j) =2.0d0*pi*ran
end do
end do

do j=0,ny/2
do i=0,nx/2
CALL RANDOM_NUMBER(ran)
eta(i,j) =2.0d0*pi*ran
end do
end do

!Random phase
do j=0,ny-1
do i=0,nx-1
phase2d(1,i,j)       = 0.0d0
phase2d(2,i,j)       = 0.0d0
end do
end do
  
do j=1,ny/2-1
do i=1,nx/2-1
!I.st
phase2d(1,i,j)       = dcos(ksi(i,j)+eta(i,j)) 
phase2d(2,i,j)       = dsin(ksi(i,j)+eta(i,j)) 
!II.nd
phase2d(1,nx-i,j)    = dcos(-ksi(i,j)+eta(i,j)) 
phase2d(2,nx-i,j)    = dsin(-ksi(i,j)+eta(i,j)) 
!IV.th
phase2d(1,i,ny-j)    = dcos(ksi(i,j)-eta(i,j)) 
phase2d(2,i,ny-j)    = dsin(ksi(i,j)-eta(i,j)) 
!III.rd
phase2d(1,nx-i,ny-j) = dcos(-ksi(i,j)-eta(i,j)) 
phase2d(2,nx-i,ny-j) = dsin(-ksi(i,j)-eta(i,j)) 
end do
end do


!vorticity amplitudes in Fourier space 
k=1
do j=0,ny-1
do i=0,nx-1   
    kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
    data1d(k)   =  dsqrt(kk*E4(kk)/pi)*phase2d(1,i,j)
	data1d(k+1) =  dsqrt(kk*E4(kk)/pi)*phase2d(2,i,j)   
k = k + 2
end do
end do

!find the velocity in physical space
!forward fourier transform
isign= 1
call fourn(data1d,nn,ndim,isign)


k=1
do j=0,ny-1
do i=0,nx-1
ww(i,j)=data1d(k)
k=k+2
end do
end do


! periodicity
do j=0,ny-1
ww(nx,j)=ww(0,j)
end do
do i=0,nx
ww(i,ny)=ww(i,0)
end do

!back to the local grid
nx = ni
ny = nj
do j=0,ny
do i=0,nx
w(i,j)=ww(i*ii,j*jj)
end do
end do

deallocate(data1d,phase2d,ksi,eta,ww)


do j = 0,ny-1
do i = 0,nx-1
w_org(i,j) = w(i,j)
end do
end do

deallocate(w)


return
end


!---------------------------------------------------------------------------!
!Given energy spectrum
!Used for initial field calculation for 2D HIT
!Validated
!---------------------------------------------------------------------------!
double precision function E4(kr)
implicit none
double precision:: kr,pi,c,k0
k0 = 10.0d0
pi = 4.0d0*datan(1.0d0)
c = 4.0d0/(3.0d0*dsqrt(pi)*(k0**5))
!c = 1.0d0/(4.0d0*pi*(k0**6))
!c = 1.0d0/(2.0d0*pi*(k0**6))
E4 = c*(kr**4)*dexp(-(kr/k0)**2)
end



!-----------------------------------------------------------------!
! fft routine from numerical recipes
! ndim: dimension of the transform (i.e.; 2 for 2d problems)
! nn  : number of points in each direction
! data: one-dimensional array including real and imaginary part 
!-----------------------------------------------------------------!
subroutine fourn(data,nn,ndim,isign)
implicit none
integer:: ndim,isign
integer:: nn(ndim)
real*8:: data(*)
real*8:: wr,wi,wpr,wpi,wtemp,theta,tempr,tempi
integer::ntot,n,nrem,nprev,idim,ip1,ip2,ip3,i1,i2,i3
integer::i2rev,i3rev,ibit,ifp1,ifp2,k1,k2

      ntot=1
      do 11 idim=1,ndim
        ntot=ntot*nn(idim)
11    continue
      nprev=1
      do 18 idim=1,ndim
        n=nn(idim)
        nrem=ntot/(n*nprev)
        ip1=2*nprev
        ip2=ip1*n
        ip3=ip2*nrem
        i2rev=1
        do 14 i2=1,ip2,ip1
          if(i2.lt.i2rev)then
            do 13 i1=i2,i2+ip1-2,2
              do 12 i3=i1,ip3,ip2
                i3rev=i2rev+i3-i2
                tempr=data(i3)
                tempi=data(i3+1)
                data(i3)=data(i3rev)
                data(i3+1)=data(i3rev+1)
                data(i3rev)=tempr
                data(i3rev+1)=tempi
12            continue
13          continue
          endif
          ibit=ip2/2
1         if ((ibit.ge.ip1).and.(i2rev.gt.ibit)) then
            i2rev=i2rev-ibit
            ibit=ibit/2
          go to 1
          endif
          i2rev=i2rev+ibit
14      continue
        ifp1=ip1
2       if(ifp1.lt.ip2)then
          ifp2=2*ifp1
          theta=isign*6.28318530717959d0/(ifp2/ip1)
          wpr=-2.d0*dsin(0.5d0*theta)**2
          wpi=dsin(theta)
          wr=1.d0
          wi=0.d0
          do 17 i3=1,ifp1,ip1
            do 16 i1=i3,i3+ip1-2,2
              do 15 i2=i1,ip3,ifp2
                k1=i2
                k2=k1+ifp1
                tempr=sngl(wr)*data(k2)-sngl(wi)*data(k2+1)
                tempi=sngl(wr)*data(k2+1)+sngl(wi)*data(k2)
                data(k2)=data(k1)-tempr
                data(k2+1)=data(k1+1)-tempi
                data(k1)=data(k1)+tempr
                data(k1+1)=data(k1+1)+tempi
15            continue
16          continue
            wtemp=wr
            wr=wr*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi
17        continue
          ifp1=ifp2
        go to 2
        endif
        nprev=n*nprev
18    continue

return
end

!-----------------------------------------------------------------!
!Compute energy spectra and send back to python - postprocessing
!Validated
!-----------------------------------------------------------------!
subroutine spec(w_org,nx,ny,eplot,n)
implicit none
integer,intent(in) ::nx,ny,n
double precision, intent(in)::w_org(0:nx-1,0:ny-1)
double precision::pi
double precision,dimension(0:n),intent(inout) :: eplot
integer::i,j,k,ic
double precision::kx(0:nx-1),ky(0:ny-1),kk
double precision,parameter:: tiny=1.0d-10
double precision,dimension(:),allocatable:: data1d
double precision,dimension(:,:),allocatable::es
integer,parameter::ndim=2
integer::nn(ndim),isign

allocate(data1d(2*nx*ny))

pi = 4.0d0*datan(1.0d0)

nn(1)= nx
nn(2)= ny

!finding fourier coefficients of w 
!invese fourier transform
!find the vorticity in Fourier space
k=1
do j=0,ny-1  
do i=0,nx-1   
  data1d(k)   =  w_org(i,j)
  data1d(k+1) =  0.0d0    
k = k + 2
end do
end do
!normalize
do k=1,2*nx*ny
data1d(k)=data1d(k)/dfloat(nx*ny)
end do
!inverse fourier transform
isign= -1
call fourn(data1d,nn,ndim,isign)


!Wave numbers 
do i=0,nx/2-1
kx(i)      = dfloat(i)
kx(i+nx/2) = dfloat(i-nx/2)
end do
kx(0) = tiny

do j=0,ny/2-1
ky(j)      = dfloat(j)
ky(j+ny/2) = dfloat(j-ny/2)
end do
ky(0) = tiny

!Energy spectrum (for all wavenumbers)
allocate(es(0:nx-1,0:ny-1))
k=1
do j=0,ny-1
do i=0,nx-1 
kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
es(i,j) = pi*(data1d(k)*data1d(k) + data1d(k+1)*data1d(k+1))/kk
k = k + 2
end do
end do

!Plot angle averaged energy spectrum
do k=1,n
eplot(k) = 0.0d0
ic = 0
do j=1,ny-1
do i=1,nx-1
kk = dsqrt(kx(i)*kx(i) + ky(j)*ky(j))
    if(kk.ge.(dfloat(k)-0.5d0).and.kk.le.(dfloat(k)+0.5d0)) then
    ic = ic + 1
    eplot(k) = eplot(k) + es(i,j)
    end if
end do
end do
eplot(k) = eplot(k) / dfloat(ic)
end do

deallocate(data1d,es)

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
