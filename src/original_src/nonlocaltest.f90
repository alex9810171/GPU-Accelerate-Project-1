program nonlocal_2D_xy
implicit none
!using band diagonal to solve Jacobian*delrho=residue
!compressed_A(Jacobian)*X(delrho)=B(residue) 
integer::row,column
real*8 :: start, finish
real*8 :: D,k_small,dir
real*8 :: vmax,y0
real*8 :: y_domain_size,x_domain_size,deltav
real*8 :: dx,dy       
real*8 :: vy,dradius,y_coordinate
real*8 :: error,alpha,iter_time,del
real*8 :: u_excess,rhobulk,phi_surface
real*8 :: deltarho
real*8 :: delphi_deln0_bulk,delphi_deln1_bulk,delphi_deln2_bulk,delphi_deln3_bulk
integer:: x,info,i,j,k,l,m,n,o,p,iter

real*8,parameter :: pi=dacos(-1.d0)
real*8,allocatable :: rhosmall(:),rhosmall_old(:)
real*8,allocatable :: vx(:),y(:),displacement(:)
real*8,allocatable :: Jacobian(:,:),residue(:),distribution(:,:),al(:,:),ans(:)

real*8,allocatable :: n0_small(:,:),n1_small(:,:),n1x_small(:,:),n2_small(:,:),n2x_small(:,:),n3_small(:,:)
real*8,allocatable :: deltaphideltarho_small_2D(:,:),deltaphideltarho_small_p(:)
real*8,allocatable :: deltaphideltarho_small_1D(:)
real*8,allocatable :: delphi_deln0(:,:),delphi_deln1(:,:),delphi_deln1x(:,:),delphi_deln2(:,:),&
					  delphi_deln2x(:,:),delphi_deln3(:,:)
integer*8,allocatable::indx(:)
open (1, file = 'parameters.in', status ='old')
open (11,file = '.txt')
open (12,file = 'nonlocalresidue2Dxy.txt')		
open (13,file = 'nonlocaldistribution2Dxy.txt')
open (14,file = 'nonlocaljacobian2Dxy.txt')	
open (15,file = '2D_cartesian_initial.txt')	
open (16,file = 'delphidelrho2D.txt')	
open (17,file = 'delphidelrho1D.txt')
open (18,file = 'n0.txt')
open (19,file = 'n1.txt')
open (20,file = 'n1x.txt')
open (21,file = 'n2.txt')
open (22,file = 'n2x.txt')
open (23,file = 'n3.txt')
open (24,file = 'delphi_deln0.txt')
open (25,file = 'delphi_deln1.txt')
open (26,file = 'delphi_deln1x.txt')
open (27,file = 'delphi_deln2.txt')
open (28,file = 'delphi_deln2x.txt')
open (29,file = 'delphi_deln3.txt')			
read (1,*) y_domain_size			! set dx equal dy
read (1,*) x_domain_size
read (1,*) dy
read (1,*) dx
read (1,*) D
read (1,*) k_small
read (1,*) vmax
read (1,*) rhobulk
read (1,*) deltav
read (1,*) iter_time

close(1) 

m=int(x_domain_size/dx)
n=int(y_domain_size/dy)
dradius=dy
call cpu_time(start)
!allocate the Jacobianrix
allocate(vx((m+1)*(n+1)),y((m+1)*(n+1)))
allocate(Jacobian(((m+1)*(n+1)),(2*n+3)))
allocate(rhosmall((m+1)*(n+1)))
allocate(rhosmall_old((m+1)*(n+1)))

allocate(displacement((m+1)*(n+1)))
allocate(residue((m+1)*(n+1)))
allocate(distribution((n+1),(m+1)),indx(((m+1)*(n+1))))
allocate(al((n+1)*(m+1),(n+1)))
allocate(n0_small(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),n1_small(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),&
	n1x_small(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),n2_small(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1)&
	,n2x_small(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),n3_small(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1))



allocate(delphi_deln0(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),delphi_deln1(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1)&
		,delphi_deln1x(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),delphi_deln2(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),&
			delphi_deln2x(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1),delphi_deln3(-nint(.5d0/dradius):n+nint(.5d0/dradius),m+1))	
allocate(deltaphideltarho_small_2D((n+1),(m+1)))
allocate(deltaphideltarho_small_1D((m+1)*(n+1)))


deltaphideltarho_small_2D = 0.d0	

!set vx
do row=1,((m+1)*(n+1))
	y(row)=dy*real(row-1.d0)
	do while (y(row) .gt. y_domain_size)
		y(row)=y(row-(n+1))
	end do

	vx(row)=vmax*(1.d0-(y(row)-deltav)*(y(row)-deltav)/deltav/deltav)
end do
vy=0.d0

!Caculate bulk region properties
call dphi_dn0(rhobulk*pi/6.d0,delphi_deln0_bulk)
call dphi_dn1(rhobulk*pi,&
		rhobulk*pi/6.d0,delphi_deln1_bulk)
call dphi_dn2(rhobulk*.5d0,rhobulk*pi&
		,0.d0,rhobulk*pi/6.d0,delphi_deln2_bulk)
call dphi_dn3(rhobulk,rhobulk*.5d0,0.d0&
		,rhobulk*pi,0.d0,rhobulk*pi/6.d0&
		,delphi_deln3_bulk)
		
		phi_surface=pi/6.d0*rhobulk
		u_excess=phi_surface*(8.d0-9.d0*phi_surface+3.d0*phi_surface**2.d0)/(1.d0-phi_surface)**3.d0

		residue=rhobulk

! now build the Jacobianrix band jacobian((n+1)(m+1),(n+1)(m+1))
do row=0,(n+1)*(m+1)-1
	read(15,*)rhosmall(row)
end do
	! rhosmall=0.d0
	error=1.d0
	iter=0.d0
	del=1.d-6
	residue=0.d0
	! write(*,*) "---Calculating small weighted density---"	

! do while (iter .lt. 20)
do while (error .gt. 1.d-6)
				n0_small=0.d0
			n1_small=0.d0
			n1x_small=0.d0
			n2_small=0.d0
			n2x_small=0.d0
			n3_small=0.d0		



	do o=1,m+1





	

		do k=-nint(.5d0/dradius),n+nint(.5d0/dradius)

			y0=real(k)*dy
			do l=-nint(.5d0/dradius),nint(.5d0/dradius)
					y_coordinate=y0+l*dradius
				if(nint(y_coordinate/dy).lt.0) then
					n2_small(k,o)=n2_small(k,o)+0.d0
					n2x_small(k,o)=n2x_small(k,o)+0.d0
					! n2x_small_prime(k)=n2x_small_prime(k)+0.d0
					n3_small(k,o)=n3_small(k,o)+0.d0	

				elseif(nint(y_coordinate/dy).gt.n)then
					n2_small(k,o)=n2_small(k,o)+rhobulk*pi*dradius
					n2x_small(k,o)=n2x_small(k,o)+0.d0
					! n2x_small_prime(k)=n2x_small_prime(k)+0.d0
					n3_small(k,o)=n3_small(k,o)+rhobulk*pi*((.5d0)**2.d0-(l*dradius)**2.d0)*dradius				

				else
					n2_small(k,o)=n2_small(k,o)&
					+rhosmall(nint(y_coordinate/dy)+(o-1)*(n+1)+1)&
					*pi*dradius
					n2x_small(k,o)=n2x_small(k,o)&
					-rhosmall(nint(y_coordinate/dy)+(o-1)*(n+1)+1)&
					*pi*dradius*(l*dradius)/.5d0
					! n2x_small_prime(k)=n2x_small_prime(k)&
					! -(rho_small(nint(y_coordinate/dy))+deltarho)&
					! *2.d0*pi*sqrt((.5d0)**2.d0-(l*dradius)**2.d0)*dradius&
					! *(l*dradius)/.5d0
					n3_small(k,o)=n3_small(k,o)&
					+rhosmall(nint(y_coordinate/dy)+(o-1)*(n+1)+1)*pi*((.5d0)**2.d0-(l*dradius)**2.d0)*dradius

				endif

			! if(isnan(residue(nint(y_coordinate/dy))))then
				! write(*,*) k,l,residue(10),"n2 is nan"
				! goto 60
			! elseif(isnan(n2x_small(k)))then
				! write(*,*) k,l,"n2x is nan"
				! goto 60	
			! elseif(isnan(n3_small(k)))then	
				! write(*,*) k,l,"n3 is nan"
				! goto 60	
			! endif				
			enddo

		enddo	

			! n1x_small_prime=n2x_small_prime/(4.d0*pi*.5d0)
	enddo
	do o=1,m+1

		do k=-nint(.5d0/dradius),n+nint(.5d0/dradius)
			n0_small(k,o)=n2_small(k,o)/(4.d0*pi*.5d0**2.d0)
			n1_small(k,o)=n2_small(k,o)/(4.d0*pi*.5d0)
			n1x_small(k,o)=n2x_small(k,o)/(4.d0*pi*.5d0)
		end do
	end do
do i=-nint(.5d0/dradius),n+nint(.5d0/dradius)
	write(18,*)(n0_small(i,j),j=1,m+1)
	write(19,*)(n1_small(i,j),j=1,m+1)
	write(20,*)(n1x_small(i,j),j=1,m+1)
	write(21,*)(n2_small(i,j),j=1,m+1)
	write(22,*)(n2x_small(i,j),j=1,m+1)
	
	write(23,*)(n3_small(i,j),j=1,m+1)
	! if (i ==21) then
	! write(*,*)(n3_small(i,j),j=1,m+1)
	! end if
end do

do o=1,m+1			
		do k=-nint(.5d0/dradius),n+nint(.5d0/dradius)

			call dphi_dn0(n3_small(k,o),delphi_deln0(k,o))
			! if (k ==21) then
			! write(*,*)n3_small(k,o),delphi_deln0(k,o)
			! end if
			call dphi_dn1(n2_small(k,o),n3_small(k,o),delphi_deln1(k,o))
			call dphi_dn1x(n2x_small(k,o),n3_small(k,o),delphi_deln1x(k,o))
			call dphi_dn2(n1_small(k,o),n2_small(k,o),n2x_small(k,o),n3_small(k,o),delphi_deln2(k,o))
			call dphi_dn2x(n1x_small(k,o),n2_small(k,o),n2x_small(k,o),n3_small(k,o),delphi_deln2x(k,o))
			call dphi_dn3(n0_small(k,o),n1_small(k,o),n1x_small(k,o),n2_small(k,o),n2x_small(k,o)&
									,n3_small(k,o),delphi_deln3(k,o))							
		enddo	
			! write(*,*) "---Calculating small functional derivative---"
			
			! deltaphideltarho_small_2D_p = 0.d0
end do
do i=-nint(.5d0/dradius),n+nint(.5d0/dradius)
	write(24,*)(delphi_deln0(i,j),j=1,m+1)
	write(25,*)(delphi_deln1(i,j),j=1,m+1)
	write(26,*)(delphi_deln1x(i,j),j=1,m+1)
	write(27,*)(delphi_deln2(i,j),j=1,m+1)
	write(28,*)(delphi_deln2x(i,j),j=1,m+1)
	write(29,*)(delphi_deln3(i,j),j=1,m+1)
end do

deltaphideltarho_small_2D=0.d0
do o=1,m+1	
		
		do k=1,n+1
			y0=real(k-1)*dy
			do l=-nint(.5d0/dradius),nint(.5d0/dradius)
				y_coordinate=y0+l*dradius	
			! if(nint(y_coordinate/dy).gt.0.d0) then
				if(nint(y_coordinate/dy).gt.n) then
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln0_bulk&
						*pi*dradius/pi
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln1_bulk&
						*pi*dradius/(4.d0*.5d0*pi)
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln2_bulk&
						*pi*dradius			
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln3_bulk&
						*pi*((.5d0)**2.d0-(l*dradius)**2.d0)*dradius
				else
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln0(nint(y_coordinate/dy),o)&
						*pi*dradius/pi
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln1(nint(y_coordinate/dy),o)&
						*pi*dradius/(4.d0*.5d0*pi)
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln1x(nint(y_coordinate/dy),o)&
						*pi*dradius*(l*dradius)/.5d0/(4.d0*.5d0*pi)	
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln2(nint(y_coordinate/dy),o)&
						*pi*dradius
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln2x(nint(y_coordinate/dy),o)&
						*pi*dradius*(l*dradius)/.5d0					
					deltaphideltarho_small_2D(k,o)=deltaphideltarho_small_2D(k,o)&
						+delphi_deln3(nint(y_coordinate/dy),o)&
						*pi*((.5d0)**2.d0-(l*dradius)**2.d0)*dradius		
				endif
			! endif
			enddo	
		enddo
end do
do i=1,n+1
	write(16,*)(deltaphideltarho_small_2D(i,j),j=1,m+1)
end do


deltaphideltarho_small_1D=0.d0
do i=1,n+1
	do j=1,m+1
	deltaphideltarho_small_1D((j-1)*(n+1)+i)=deltaphideltarho_small_2D(i,j)
	
	end do
end do
do i=1,(n+1)*(m+1)
	write(17,*)deltaphideltarho_small_1D(i)
end do
if (mod(iter,5)==0) then

write(*,*)iter,error

end if
Jacobian=0.d0
! deltaphideltarho_small_1D=0.d0
	do row=1,(n+1)*(m+1)

		if((row .le. (n+1)))then	

			Jacobian(row,n+2)=1.d0
		elseif((row .gt. (n+1)) .and. (row .le. (n+1)*m))then
			
			if(int(mod(row,(n+1)))/=1 .and. int(mod(row,(n+1)))/= 0)then
				Jacobian(row,1)=(f ((rhosmall(row-(n+1))+del),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,n+1)=(f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)+del),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,n+2)=(f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row)+del)&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,n+3)=(f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)+del),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,2*n+3)=(f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))+del),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
			else
				Jacobian(row,n+2)=1.d0	
			end if
		else
			if(int(mod(row,(n+1)))/=1 .and. int(mod(row,(n+1)))/= 0)then
				Jacobian(row,1)=(fb ((rhosmall(row-(n+1))+del),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,n+1)=(fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)+del),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,n+2)=(fb ((rhosmall(row-(n+1))),(rhosmall(row)+del)&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
				Jacobian(row,n+3)=(fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)+del),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))&
				-fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1))))/del
				
			else
				Jacobian(row,n+2)=1.d0	
			end if
			
		end if

	end do
do row=1,(n+1)*(m+1)
	write(11,*)(Jacobian(row,column),column=1,2*n+3)
end do				

	!print out discretized_Jacobianrix

	!then build the Jacobianrix residue(right-hand-side)
	do row=1,((m+1)*(n+1))
		if((row .le. (n+1)))then	
			residue(row)=rhosmall(row)-rhobulk
			
		elseif((row .gt. (n+1)) .and. (row .le. (n+1)*m))then
			
			if(int(mod(row,(n+1)))/=1 .and. int(mod(row,(n+1)))/= 0)then
				residue(row)=f ((rhosmall(row-(n+1))),(rhosmall(row+(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row+(n+1))),(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))
			elseif (int(mod(row,(n+1)))==1 )then
				residue(row)=rhosmall(row)
			else
				residue(row)=rhosmall(row)-rhobulk
			end if
		else
			if(int(mod(row,(n+1)))/=1 .and. int(mod(row,(n+1)))/= 0)then
				residue(row)=fb ((rhosmall(row-(n+1))),(rhosmall(row))&
				,(rhosmall(row+1)),(rhosmall(row-1)),(deltaphideltarho_small_1D(row-(n+1)))&
				,(deltaphideltarho_small_1D(row))&
				,(deltaphideltarho_small_1D(row+1)),(deltaphideltarho_small_1D(row-1)))			
				
			elseif (int(mod(row,(n+1)))==1 )then
				residue(row)=rhosmall(row)
			else
				residue(row)=rhosmall(row)-rhobulk
			end if
		end if
		! if((row .le. (n+1)))then													!x-dir boundary condition
			! residue(row)=rhobulk
		! elseif (int(mod(row,(n+1)))/=1 .and. int(mod(row,(n+1)))/= 0)then			!those points inside the grid 
			! residue(row)=
		! elseif (int(mod(row,(n+1)))== 1)then											!boundary condition on the plate
			! residue(row)=0.d0
		! elseif (int(mod(row,(n+1)))== 0)then											!boundary condition far from plate
			! residue(row)=rhobulk
		! end if
	end do 
	! if (mod(column,10)==0) then
		! write
	!print out right hand side
	do row=1,(n+1)*(m+1)	
		write(12,*)(residue(row))
	end do

	!finale!!now just call gaussj

	call bandec(Jacobian,n+1,n+1,al,indx,d)

	call banbks(Jacobian,n+1,n+1,al,indx,residue)


	do row=1,(n+1)*(m+1)
		rhosmall_old(row)=rhosmall(row)
		displacement(row)=residue(row)
		rhosmall(row)=rhosmall(row)-displacement(row)

	end do
	! print out the concentration distribution
	error=maxval(abs(rhosmall_old-rhosmall))	
	iter=iter+1

enddo

do row=1,(n+1)*(m+1)

	distribution(mod(row,(n+1)),(row/(n+1))+1)=rhosmall(row)
	

end do

do i=1,n+1
	write(13,*)(distribution(i,j),j=1,m+1)

end do

call cpu_time(finish)
write(*,*)error
write(*,*)finish-start,iter-1
contains
FUNCTION f (rholeft,rhoright,rhomid,rhoup,rhodown,delphidelrholeft&
	,delphidelrhoright,delphidelrhomid,delphidelrhoup,delphidelrhodown)
	  real*8::f
	  real*8, intent(in)::rholeft,rhoright,rhomid,rhoup,rhodown,delphidelrholeft&
	  ,delphidelrhoright,delphidelrhomid,delphidelrhoup,delphidelrhodown
	  f=&
		vx(row)/(2.d0*dx)*(rhoright-rholeft)+&
		-D*((rhoright+rholeft&
		-2.d0*rhomid)/(dx**2)+(rhoup+rhodown+&
		-2.d0*rhomid)/(dy**2)+(rhoright-rholeft)/(2*dx)*(delphidelrhoright-delphidelrholeft)/(2*dx)+&
		(rhoup-rhodown)/(2*dy)*(delphidelrhoup-delphidelrhodown)/(2*dy))
	  END FUNCTION f
	FUNCTION fb (rholeft,rhomid,rhoup,rhodown,delphidelrholeft,&
	delphidelrhomid,delphidelrhoup,delphidelrhodown)
	  real*8::fb
	  real*8, intent(in)::rholeft,rhomid,rhoup,rhodown,delphidelrholeft&
	  ,delphidelrhomid,delphidelrhoup,delphidelrhodown
	  fb=&

		vx(row)/(2.d0*dx)*(rhomid-rholeft)+&
		-D*((rholeft&
		-rhomid)/(dx**2)+(rhoup+rhodown+&
		-2.d0*rhomid)/(dy**2)+(rhomid-rholeft)/(2*dx)*(delphidelrhomid-delphidelrholeft)/(2*dx)+&
		(rhoup-rhodown)/(2*dy)*(delphidelrhoup-delphidelrhodown)/(2*dy))
	  END FUNCTION fb

subroutine dphi_dn0(n3,output)
implicit none
real*8,intent(in) :: n3
real*8,intent(out):: output
output=-log(1.d0-n3)
return 
end subroutine dphi_dn0

subroutine dphi_dn1(n2,n3,output)
implicit none
real*8,intent(in) :: n2,n3
real*8,intent(out):: output
output=n2/(1.d0-n3)
return
end subroutine dphi_dn1

subroutine dphi_dn1x(n2x,n3,output)
implicit none
real*8,intent(in) :: n2x,n3
real*8,intent(out):: output
output=-n2x/(1.d0-n3)
return
end subroutine dphi_dn1x

subroutine dphi_dn2(n1,n2,n2x,n3,output)
implicit none
real*8,intent(in) :: n1,n2,n2x,n3
real*8,intent(out):: output
real*8:: pi
parameter (pi=dacos(-1.d0))
output= n1/(1.d0-n3)+(3.d0*n2**2.d0-3.d0*n2x**2.d0)&
	/(24.d0*pi*(1.d0-n3)**2.d0)
return
end subroutine dphi_dn2

subroutine dphi_dn2x(n1x,n2,n2x,n3,output)
implicit none
real*8,intent(in) :: n1x,n2,n2x,n3
real*8,intent(out):: output
real*8:: pi
parameter (pi=dacos(-1.d0))
output= -n1x/(1.d0-n3)+(-3.d0*n2*2.d0*n2x)&
	/(24.d0*pi*(1.d0-n3)**2.d0)
return
end subroutine dphi_dn2x

subroutine dphi_dn3(n0,n1,n1x,n2,n2x,n3,output)
implicit none
real*8,intent(in) :: n0,n1,n1x,n2,n2x,n3
real*8,intent(out):: output
real*8:: pi
parameter (pi=dacos(-1.d0))
output= n0/(1.d0-n3)+(n1*n2-n1x*n2x)/(1.d0-n3)**2.d0&
	+2.d0*(n2**3.d0-3.d0*n2*n2x**2.d0)/(24.d0*pi*(1.d0-n3)**3.d0)
return
end subroutine dphi_dn3
SUBROUTINE bandec(a,m1,m2,al,indx,d)
USE nrtype; USE nrutil, ONLY : assert_eq,imaxloc,swap,arth
IMPLICIT NONE
REAL(DP), DIMENSION(:,:), INTENT(INOUT) :: a
INTEGER(I4B), INTENT(IN) :: m1,m2
REAL(DP), DIMENSION(:,:), INTENT(OUT) :: al
INTEGER(I8B), DIMENSION(:), INTENT(OUT) :: indx
REAL(DP), INTENT(OUT) :: d
REAL(DP), PARAMETER :: TINY=1.0e-20_dp
! Given an N × N band diagonal matrix A with m1 subdiagonal rows and m2 superdiagonal
! rows, compactly stored in the array a(1:N,1:m1+m2+1) as described in the comment for
! routine banmul, this routine constructs an LU decomposition of a rowwise permutation of
! A. The upper triangular matrix replaces a, while the lower triangular matrix is returned in
! al(1:N,1:m1). indx is an output vector of length N that records the row permutation
! effected by the partial pivoting; d is output as ±1 depending on whether the number of
! row interchanges was even or odd, respectively. This routine is used in combination with
! banbks to solve band-diagonal sets of equations.
INTEGER(I4B) :: i,k,l,mdum,mm,n
REAL(DP) :: dum
n=assert_eq(size(a,1),size(al,1),size(indx),'bandec: n')
mm=assert_eq(size(a,2),m1+m2+1,'bandec: mm')
mdum=assert_eq(size(al,2),m1,'bandec: mdum')
a(1:m1,:)=eoshift(a(1:m1,:),dim=2,shift=arth(m1,-1,m1)) 		!Rearrange the storage a
d=1.0 															!bit.
do k=1,n 														!For each row...
l=min(m1+k,n)
i=imaxloc(abs(a(k:l,1)))+k-1 									!Find the pivot element.
dum=a(i,1)
if (dum == 0.0) a(k,1)=TINY
! Matrix is algorithmically singular, but proceed anyway with TINY pivot (desirable in some
! applications).
indx(k)=i
if (i /= k) then 												!Interchange rows.
d=-d
call swap(a(k,1:mm),a(i,1:mm))
end if
do i=k+1,l 														!Do the elimination.
dum=a(i,1)/a(k,1)
al(k,i-k)=dum
a(i,1:mm-1)=a(i,2:mm)-dum*a(k,2:mm)
a(i,mm)=0.0
end do
end do
END SUBROUTINE bandec

SUBROUTINE banbks(a,m1,m2,al,indx,b)
USE nrtype; USE nrutil, ONLY : assert_eq,swap
IMPLICIT NONE
REAL(DP), DIMENSION(:,:), INTENT(IN) :: a,al
INTEGER(I4B), INTENT(IN) :: m1,m2
INTEGER(I8B), DIMENSION(:), INTENT(IN) :: indx
REAL(DP), DIMENSION(:), INTENT(INOUT) :: b
! Given the arrays a, al, and indx as returned from bandec, and given a right-hand-side
! vector b, solves the band diagonal linear equations A·x = b. The solution vector x overwrites
! b. The other input arrays are not modified, and can be left in place for successive calls with
! different right-hand sides.
INTEGER(I4B) :: i,k,l,mdum,mm,n
n=assert_eq(size(a,1),size(al,1),size(b),size(indx),'banbks: n')
mm=assert_eq(size(a,2),m1+m2+1,'banbks: mm')
mdum=assert_eq(size(al,2),m1,'banbks: mdum')
do k=1,n 														!Forward substitution, unscrambling the permuted rows as we
l=min(n,m1+k) 													!go.
i=indx(k)
if (i /= k) call swap(b(i),b(k))
b(k+1:l)=b(k+1:l)-al(k,1:l-k)*b(k)
end do
do i=n,1,-1 													!Backsubstitution.
l=min(mm,n-i+1)
b(i)=(b(i)-dot_product(a(i,2:l),b(1+i:i+l-1)))/a(i,1)
end do
END SUBROUTINE banbks
end program
