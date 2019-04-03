!-----------------------------------------------------------------------------------!
!-----------------------------------------------------------------------------------!
!This set of subroutines not used explicitly in ML studies - just for archival
!-----------------------------------------------------------------------------------!
!-----------------------------------------------------------------------------------!


!-----------------------------------------------------------------------------------!
!Filtering (Pade) - fourth-order Pade relaxation filtering - refer any of our papers for standard def
!Taken from NS2D.f90
!Validated
!-----------------------------------------------------------------------------------!
subroutine filter_pade(q_org,nx,ny,afil)
implicit none
integer,intent(in)::nx,ny
double precision, intent(inout) :: q_org(0:nx-1,0:ny-1)
double precision, intent(in) :: afil

integer ::i,j,k
double precision,allocatable:: q(:,:),g(:,:),a(:),b(:)

!Temp arrays
allocate(q(0:nx,0:ny))
allocate(g(0:nx,0:ny))

do j = 0,ny-1
	do i = 0,nx-1
		q(i,j) = q_org(i,j)
	end do
end do

!Periodic BC Update
do j = 0,ny-1
	q(nx,j) = q(0,j)
end do

do i = 0,nx
	q(i,ny) = q(i,0)
end do

! filter in x direction (periodic)
allocate(a(0:nx),b(0:nx))
do j=0,ny
	do i=0,nx
	a(i) = q(i,j)
	end do
		call filterPade4p(nx,a,b,afil)
	do i=0,nx
	g(i,j) = b(i)
	end do
end do
deallocate(a,b)


! filter in y direction (periodic)
allocate(a(0:ny),b(0:ny))
do i=0,nx
	do j=0,ny
	a(j) = g(i,j)
	end do
		call filterPade4p(ny,a,b,afil)
	do j=0,ny
	q(i,j) = b(j)
	end do
end do
deallocate(a,b)

do j = 0,ny-1
	do i = 0,nx-1
		q_org(i,j) = q(i,j)
	end do
end do

deallocate(q,g)

return 
end


!---------------------------------------------------------------------------!
!Compute filtered variable for a given grid data
!Pade forth-order:  -0.5 < afil < 0.5
!Periodic - validated by Dr. San
!---------------------------------------------------------------------------!
subroutine filterPade4p(n,u,uf,afil)
implicit none
integer::n,i
double precision ::afil
double precision ::u(0:n),uf(0:n)
double precision ::alpha,beta
double precision, dimension (0:n-1):: a,b,c,r,x 

do i=0,n-1
a(i) = afil
b(i) = 1.0d0
c(i) = afil
end do

do i=2,n-2
r(i) = (0.625d0 + 0.75d0*afil)*u(i) &
      +(0.5d0 + afil)*0.5d0*(u(i-1)+u(i+1))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(i-2)+u(i+2))
end do

r(1) = (0.625d0 + 0.75d0*afil)*u(1) &
      +(0.5d0 + afil)*0.5d0*(u(0)+u(2))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(n-1)+u(3))

r(0) = (0.625d0 + 0.75d0*afil)*u(0) &
      +(0.5d0 + afil)*0.5d0*(u(n-1)+u(1))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(n-2)+u(2))

r(n-1) = (0.625d0 + 0.75d0*afil)*u(n-1) &
      +(0.5d0 + afil)*0.5d0*(u(n-2)+u(n))  &
      +(-0.125d0 + 0.25d0*afil)*0.5d0*(u(n-3)+u(1))      
      
     
alpha = afil
beta  = afil

call ctdms(a,b,c,alpha,beta,r,x,0,n-1) 

do i=0,n-1
uf(i)=x(i)
end do
uf(n)=uf(0)

return 
end


!-------------------------------------------------------------------!
! solution of cyclic tridiagonal systems (periodic tridiagonal)
! n:matrix size (starting from 1)
! a:subdiagonal, b: diagonal, c:superdiagonal
! r:rhs, x:results
! alpha:sub entry (first value in e-th eq.)
! beta: super entry (last value in s-th eq.)
! Validated - Dr. San
!-------------------------------------------------------------------!
subroutine ctdms(a,b,c,alpha,beta,r,x,s,e) 
implicit none
integer:: s,e
double precision :: alpha,beta,a(s:e),b(s:e),c(s:e),r(s:e),x(s:e)  
integer:: i  
double precision :: fact,gamma,bb(s:e),u(s:e),z(s:e)
if((e-s).le.2) then
write(*,*) ' matrix too small in cyclic' 
stop
end if 
gamma=-b(s)  
bb(s)=b(s)-gamma  
bb(e)=b(e)-alpha*beta/gamma  
do i=s+1,e-1  
bb(i)=b(i)  
end do  
call tdms(a,bb,c,r,x,s,e) 	      
u(s)=gamma  
u(e)=alpha  
do i=s+1,e-1  
u(i)=0.0d0  
end do  
call tdms(a,bb,c,u,z,s,e) 
fact=(x(s)+beta*x(e)/gamma)/(1.0d0+z(s)+beta*z(e)/gamma)  
do i=s,e  
x(i)=x(i)-fact*z(i)  
end do  
return  
end


!----------------------------------------------------!
! solution tridiagonal systems (regular tri-diagonal)
! a:subdiagonal, b: diagonal, c:superdiagonal
! r:rhs, u:results
! s: starting index
! e: ending index
! Validated - Dr. San
!----------------------------------------------------!
subroutine tdms(a,b,c,r,u,s,e)
implicit none 
integer::s,e
double precision ::a(s:e),b(s:e),c(s:e),r(s:e),u(s:e) 
integer::j  
double precision ::bet,gam(s:e) 
bet=b(s)  
u(s)=r(s)/bet  
do j=s+1,e  
gam(j)=c(j-1)/bet  
bet=b(j)-a(j)*gam(j)  
u(j)=(r(j)-a(j)*u(j-1))/bet  
end do  
do j=e-1,s,-1  
u(j)=u(j)-gam(j+1)*u(j+1)  
end do  
return  
end  


!-----------------------------------------------------------------------------------!
!Filtering (Gaussian) - using definition in classical image processing literature
!Same definition as in Approximate_Deconvolution.f95
!Validated
!-----------------------------------------------------------------------------------!
subroutine filter_gaussian(q_org,nx,ny,sigma)
implicit none
integer,intent(in)::nx,ny
double precision, intent(inout) :: q_org(0:nx-1,0:ny-1)
double precision, intent(in) :: sigma

integer ::i,j,k
double precision,allocatable:: q(:,:),u(:,:),v(:,:)
double precision :: sumval,pi
double precision,dimension(-3:3) :: kernel

pi = datan(1.0d0)*4.0d0

!Obtain kernel for convolution from classical Gaussian used in image processing literature
sumval = 0.0d0
do i = -3,3
kernel(i) = dexp(-(dfloat(i*i)/(2.0d0*sigma**2)))/dsqrt(2.0d0*(sigma**2)*pi)
sumval = sumval + kernel(i)
end do

!Normalize the kernel to unity
do i =-3,3
kernel(i) = kernel(i)/sumval
end do

allocate(q(-3:nx+2,-3:ny+2))
allocate(u(-3:nx+2,-3:ny+2))
allocate(v(-3:nx+2,-3:ny+2))

do j=0,ny-1
do i=0,nx-1
u(i,j) = q_org(i,j)
end do
end do

call periodic_bc_update(nx,ny,u)

!filter in y
do j=0,ny-1
do i=0,nx-1

sumval = 0.0d0
do k = -3,3
sumval = sumval + kernel(k)*u(i,j+k)
end do
v(i,j) = sumval




end do
end do

call periodic_bc_update(nx,ny,v)

!filter in x

do j=0,ny-1
do i=0,nx-1


sumval = 0.0d0
do k = -3,3
sumval = sumval + kernel(k)*v(i,j+k)
end do
q(i,j) = sumval



                           
end do
end do
 
call periodic_bc_update(nx,ny,q)

do j = 0,ny-1
     do i = 0,nx-1
          q_org(i,j) = q(i,j)
     end do
end do

deallocate(u,v,q)

return
end


!---------------------------------------------------------------------------!
!subroutine - boundary condition update
!Validated
!---------------------------------------------------------------------------!
subroutine periodic_bc_update(nx,ny,u)
implicit none

integer :: nx, ny, i, j
double precision, dimension(-3:nx+2,-3:ny+2) :: u

do i = 0,nx-1
u(i,-1) = u(i,ny-1)
u(i,-2) = u(i,ny-2)
u(i,-3) = u(i,ny-3)


u(i,ny) = u(i,0)
u(i,ny+1) = u(i,1)
u(i,ny+2) = u(i,2)
end do

do j = -3,ny+2
u(-1,j) = u(nx-1,j)
u(-2,j) = u(nx-2,j)
u(-3,j) = u(nx-3,j)


u(nx,j) = u(0,j)
u(nx+1,j) = u(1,j)
u(nx+2,j) = u(2,j)

end do

end subroutine