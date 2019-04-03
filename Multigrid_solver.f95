
!---------------------------------------------------------------------------!
!Relaxation formula for Poisson equation
!Uses GS relaxation
!Modified for periodic BCs (-1:nx,-1:ny)
!Takenfrom MAE5093 - Github repo - validated
!---------------------------------------------------------------------------!
subroutine relax(nx,ny,dx,dy,f,u)
implicit none
integer::nx,ny
double precision ::dx,dy
double precision, dimension(-1:nx,-1:ny)::u,f
double precision ::a
integer::i,j

a = -2.0d0/(dx*dx) - 2.0d0/(dy*dy)
  
do i=0,nx-1
do j=0,ny-1
u(i,j) = (1.0d0/a)*(f(i,j) &
                   - (u(i+1,j)+u(i-1,j))/(dx*dx) &
                   - (u(i,j+1)+u(i,j-1))/(dy*dy) )
end do
end do

call periodic_bc_update(nx,ny,u)

return
end



!---------------------------------------------------------------------------!
!Residual formula for Poisson equation
!Implemented for Periodic BCs (-1:nx,-1:ny) - validated
!---------------------------------------------------------------------------!
subroutine resid(nx,ny,dx,dy,f,u,r)
!$ use omp_lib

implicit none
integer::nx,ny
double precision ::dx,dy
double precision, dimension(-1:nx,-1:ny)::u,f,r
integer::i,j


!$OMP PARALLEL DO
do i=0,nx-1
do j=0,ny-1
r(i,j) = f(i,j) - (u(i+1,j) - 2.0d0*u(i,j) + u(i-1,j))/(dx*dx) &
		        - (u(i,j+1) - 2.0d0*u(i,j) + u(i,j-1))/(dy*dy) 
end do
end do
!$OMP END PARALLEL DO


!Boundary conditions for residuals
call periodic_bc_update(nx,ny,r)

return
end

!---------------------------------------------------------------------------!
!Compute L2-norm for an array
!Validated
!---------------------------------------------------------------------------!
subroutine l2norm(nx,ny,r,rms)
implicit none
integer::nx,ny
double precision, dimension(-1:nx,-1:ny)::r
integer::i,j
double precision ::rms

rms=0.0d0
do i=0,nx-1
do j=0,ny-1
rms = rms + r(i,j)*r(i,j)
end do 
end do
rms= dsqrt(rms/dfloat((nx-1)*(ny-1)))

return
end


!---------------------------------------------------------------------------!
!Restriction operators
!Modified for periodic BCs
!Validated
!---------------------------------------------------------------------------!
subroutine rest(nxf,nyf,nxh,nyh,r,f)
!$ use omp_lib

implicit none
integer::nxf,nyf,nxh,nyh
double precision, dimension(-1:nxf,-1:nyf)::r	!on finer grid
double precision, dimension(-1:nxh,-1:nyh)::f	!on coarser grid
integer::i,j
integer::ireo

ireo = 3

if (ireo.eq.1) then !simply injection

!$OMP PARALLEL DO
do i=0,nxh-1
do j=0,nyh-1
f(i,j) = r(2*i,2*j) 							  	
end do
end do
!$OMP END PARALLEL DO

else if (ireo.eq.2) then !half-weight

!$OMP PARALLEL DO
do i=0,nxh-1
do j=0,nyh-1
f(i,j) = 1.0d0/8.0d0*( 4.0d0*r(2*i,2*j) &
	     + 1.0d0*(r(2*i+1,2*j)+r(2*i-1,2*j)+r(2*i,2*j+1)+r(2*i,2*j-1)) )							  	
end do
end do
!$OMP END PARALLEL DO

else !full-weight (trapezoidal)
!$OMP PARALLEL DO
do i=0,nxh-1
do j=0,nyh-1
f(i,j) = 1.0d0/16.0d0*( 4.0d0*r(2*i,2*j) &
	     + 2.0d0*(r(2*i+1,2*j)+r(2*i-1,2*j)+r(2*i,2*j+1)+r(2*i,2*j-1)) &
	     + 1.0d0*(r(2*i+1,2*j+1)+r(2*i-1,2*j-1)+r(2*i-1,2*j+1)+r(2*i+1,2*j-1)))							  	
end do
end do
!$OMP END PARALLEL DO
end if


call periodic_bc_update(nxh,nyh,f)

  
return
end


!---------------------------------------------------------------------------!
!Prolongation operator
!bilinear interpolation
!---------------------------------------------------------------------------!
subroutine prol(nxh,nyh,nxf,nyf,u,p)
!$ use omp_lib

implicit none
integer::nxf,nyf,nxh,nyh
double precision, dimension(-1:nxf,-1:nyf)::p	!on higher grid
double precision, dimension(-1:nxh,-1:nyh)::u	!on lower grid
integer::i,j

!$OMP PARALLEL DO
do i=0,nxh-1
do j=0,nyh-1
p(2*i,2*j)    = u(i,j)
p(2*i+1,2*j)  = 1.0d0/2.0d0*(u(i,j)+u(i+1,j))
p(2*i,2*j+1)  = 1.0d0/2.0d0*(u(i,j)+u(i,j+1))
p(2*i+1,2*j+1)= 1.0d0/4.0d0*(u(i,j)+u(i,j+1)+u(i+1,j)+u(i+1,j+1))
end do
end do
!$OMP END PARALLEL DO
call periodic_bc_update(nxf,nyf,p)

return
end


!---------------------------------------------------------------------------!
!Multigrid scheme (5 level)
!Full-weighting is used as restriction operator
!Bilinear interpolation procedure is used as prolongation operator
!Matches with GS - validated
!---------------------------------------------------------------------------!
subroutine solve_poisson_periodic(u_orig,f_orig,nx,ny,dx,dy,tol)
implicit none
integer,intent(in)::nx,ny
double precision,intent(in) ::dx,dy
double precision,intent(in) ::tol
integer::nI,v1,v2,v3
double precision,dimension(0:nx-1,0:ny-1),intent(in)::f_orig
double precision,dimension(0:nx-1,0:ny-1),intent(inout)::u_orig
double precision,dimension(:,:),allocatable::u,f
double precision,dimension(:,:),allocatable:: r,r2,r3,r4,r5
double precision,dimension(:,:),allocatable:: p,p2,p3,p4
double precision,dimension(:,:),allocatable:: u2,u3,u4,u5
double precision,dimension(:,:),allocatable:: f2,f3,f4,f5
double precision ::dx2,dy2,dx3,dy3,dx4,dy4,dx5,dy5,rms0,rms,rmsc
integer::i,j,k,me,m,nx2,ny2,nx3,ny3,nx4,ny4,nx5,ny5

allocate(u(-1:nx,-1:ny))
allocate(f(-1:nx,-1:ny))

do j = 0,ny-1
	do i = 0,nx-1
		u(i,j)=u_orig(i,j)
		f(i,j)=f_orig(i,j)
	end do
end do

call periodic_bc_update(nx,ny,u)
call periodic_bc_update(nx,ny,f)

nI = 100000 !maximum number of outer iteration
v1 = 2   	!number of relaxation for restriction in V-cycle
v2 = 2   	!number of relaxation for prolongation in V-cycle
v3 = 100 	!number of relaxation at coarsest level


dx2=dx*2.0d0
dy2=dy*2.0d0

dx3=dx*4.0d0
dy3=dy*4.0d0

dx4=dx*8.0d0
dy4=dy*8.0d0

dx5=dx*16.0d0
dy5=dy*16.0d0

nx2=nx/2
ny2=ny/2

nx3=nx/4
ny3=ny/4

nx4=nx/8
ny4=ny/8

nx5=nx/16
ny5=ny/16

me = 0

if (nx5.lt.2.or.ny5.lt.2) then
write(*,*)"5 level is high for this grid.."
stop
end if

allocate(r (-1:nx ,-1:ny))
allocate(p (-1:nx ,-1:ny))

allocate(u2(-1:nx2,-1:ny2))
allocate(f2(-1:nx2,-1:ny2))
allocate(r2(-1:nx2,-1:ny2))
allocate(p2(-1:nx2,-1:ny2))

allocate(u3(-1:nx3,-1:ny3))
allocate(f3(-1:nx3,-1:ny3))
allocate(r3(-1:nx3,-1:ny3))
allocate(p3(-1:nx3,-1:ny3))

allocate(u4(-1:nx4,-1:ny4))
allocate(f4(-1:nx4,-1:ny4))
allocate(r4(-1:nx4,-1:ny4))
allocate(p4(-1:nx4,-1:ny4))

allocate(u5(-1:nx5,-1:ny5))
allocate(f5(-1:nx5,-1:ny5))
allocate(r5(-1:nx5,-1:ny5))

!Compute initial resitual:
call resid(nx,ny,dx,dy,f,u,r)
!and its l2 norm:
call l2norm(nx,ny,r,rms0)


loop1: do k=1,nI


!1.Relax v1 times
do m=1,v1
call relax(nx,ny,dx,dy,f,u)			
end do

! Compute residual
call resid(nx,ny,dx,dy,f,u,r)

! Check for convergence on finest grid	
call l2norm(nx,ny,r,rms)

! print*,rms0,rms
! pause

if (rms/rms0.le.tol) then
	exit loop1
end if

!1r.Restriction	
call rest(nx,ny,nx2,ny2,r,f2)

!Set zero
do i=-1,nx2
do j=-1,ny2
u2(i,j)=0.0d0
end do
end do


!2.Relax v1 times
do m=1,v1
call relax(nx2,ny2,dx2,dy2,f2,u2)
end do

! Compute residual
call resid(nx2,ny2,dx2,dy2,f2,u2,r2)

!2r.Restriction
call rest(nx2,ny2,nx3,ny3,r2,f3)

!Set zero
do i=-1,nx3
do j=-1,ny3
u3(i,j)=0.0d0
end do
end do


!3.Relax v1 times
do m=1,v1
call relax(nx3,ny3,dx3,dy3,f3,u3)
end do

! Compute residual
call resid(nx3,ny3,dx3,dy3,f3,u3,r3)


!3r.Restriction
call rest(nx3,ny3,nx4,ny4,r3,f4)

!Set zero
do i=-1,nx4
do j=-1,ny4
u4(i,j)=0.0d0
end do
end do

!4.Relax v1 times
do m=1,v1
call relax(nx4,ny4,dx4,dy4,f4,u4)
end do


! Compute residual
call resid(nx4,ny4,dx4,dy4,f4,u4,r4)


!4r.Restriction
call rest(nx4,ny4,nx5,ny5,r4,f5)

!Set zero
do i=-1,nx5
do j=-1,ny5
u5(i,j)=0.0d0
end do
end do

!5.Relax v3 times (or it can be solved exactly)
!call initial residual:
	call resid(nx5,ny5,dx5,dy5,f5,u5,r5)
	call l2norm(nx5,ny5,r5,rmsc)


loop2: do m=1,v3
	call relax(nx5,ny5,dx5,dy5,f5,u5)
	call resid(nx5,ny5,dx5,dy5,f5,u5,r5)
	! Check for convergence on smallest grid	
	call l2norm(nx5,ny5,r5,rms)
	if (rms/rmsc.le.tol) then
		exit loop2
	end if
end do loop2

me = me + m

!4p.Prolongation
call prol(nx5,ny5,nx4,ny4,u5,p4)

!Correct
u4 = u4 + p4

!4.Relax v2 times
do m=1,v2
call relax(nx4,ny4,dx4,dy4,f4,u4)
end do

!3p.Prolongation
call prol(nx4,ny4,nx3,ny3,u4,p3)

!Correct
u3 = u3 + p3

!3.Relax v2 times
do m=1,v2
call relax(nx3,ny3,dx3,dy3,f3,u3)
end do

!2p.Prolongation
call prol(nx3,ny3,nx2,ny2,u3,p2)

!Correct
u2 = u2 + p2

!2.Relax v2 times
do m=1,v2
call relax(nx2,ny2,dx2,dy2,f2,u2)
end do

!1p.Prolongation
call prol(nx2,ny2,nx,ny,u2,p)


!Correct
u = u + p


!1.Relax v2 times
do m=1,v2
call relax(nx,ny,dx,dy,f,u)
end do

end do loop1! Outer iteration loop

do j = 0,ny-1
	do i = 0,nx-1
		u_orig(i,j) = u(i,j)
	end do
end do

deallocate(u,f,r,p,u2,f2,r2,p2,u3,f3,r3,p3,u4,f4,r4,p4,u5,f5,r5)

return  
end 

!---------------------------------------------------------------------------!
!subroutine - boundary condition update
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

end
