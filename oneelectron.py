#C>>>>>>>>>>> PROPAGATION IN IMAGINARY TIME: -iDT <<<<<<<<<<<<<
#C>>>>>>>>>>>>>>>>> OR REAL TIME PROPAGATION <<<<<<<<<<<<<<<<<<
#C>>>>>> PROPAGATION DEXP(-iDTV/2)DEXP(-iDTK)DEXP(-iDTV/2) <<<<<<<

import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
import scipy.special as sp
from scipy.signal import gaussian
from scipy.fftpack import fft, ifft, fftfreq

NX=1690 # NX -> # OF GRID POINTS
NORB=2 # NORB -> # OF ORBITALS
WX=1690 # WX -> SIZE OF THE SYSTEM (IN ANGSTRON)

CPSI = np.zeros((NX, NORB), dtype=np.complex_)
CPSI0 = np.zeros((NX, NORB), dtype=np.complex_)
CV = np.zeros(NX, dtype=np.complex_)
POT = np.zeros(NX)
X = np.zeros(NX)
XOR = np.zeros(NX)
XAV = np.zeros(NORB)
PSI = np.zeros(NORB)
PR = np.zeros(NORB)
DIP = np.zeros(NORB)
PROB = np.zeros(NORB)
POTT = np.zeros(NX)
XAV2 = np.zeros(NORB)

DX = 0.0
DT = 0.0
ALX = 0.0
RY = 0.0
PI = 0.0
A0 = 0.0
CEPS = 0.0
CEPS21 = 0.0
CAX = 0.0
CBX = 0.0

AM = 0.067 # GaAs
EP0 = 13.1 # GaAs

RY = 13.6058e3 * AM / EP0 / EP0
A0 = 0.5292 * EP0 / AM
PI = np.pi # 4.D0 * DATAN( 1.D0 )!!!
TWOPI = 2.D0 * PI
AHC = 1.054589 * 1.e3 / 1.602189 # mev.fs

ALX = WX / A0
ALXH = 0.5 * ALX
CZ = (0.D0,1.D0)
DX = ALX/(NX-1)

for IX in range(NX):
    X[IX] = DX * ( IX - 1 ) - ALXH

T0 = 0.0

ASIG = 0.1 * ALX                ! largura initial wave
X0 = 0.0 / A0                ! center initial wave

for IX in range(nX):
    XOR[IX] = (X[IX] - X0) / ASIG

# = PROBABIL(CPSI0,CPSI,PROB)
TIN = 0.0
#XAVG(X,CPSI,XAV,XAV2)
#      WRITE(*,*) 'DT (fsec), NT ='
#      READ(*,*) DT, NT
#      WRITE(*,*) 'V0 (meV) ='
#      READ(*,*) V0
#      V0 = V0/RY
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C======== Here we generate the initial wave-functions ========
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      IF(ICON.EQ.0) THEN
         CALL WAVEI(X0R,CPSI)
       OPEN(UNIT=30,FILE='psiINIT',FORM='FORMATTED')
       NI = 1
       NF = NORB
C       NI = 1
C       NF = 5
C2554   IF(NF.GT.NORB) NF = NORB
C       IF(NI.GT.NORB) GO TO 2553
       DO 990 IX = 1, NX, 2
        DO 90 IO = NI, NF
         PSI(IO) = DCONJG(CPSI(IX,IO)) * CPSI(IX,IO)
         IF(PSI(IO).LE.1.D-20) PSI(IO) = 0.D0
90      CONTINUE
        WRITE(30,310) X(IX)*A0,( PSI(IO)/A0, IO = NI, NF )
990    CONTINUE
C       NI = NI + 5
C       NF = NF + 5
C       GO TO 2554
C2553   CONTINUE
        END IF
C
      T = DT * NT
      TTOT = T + T0
      CDT = DT
      IF(IMR.EQ.0) CDT = -CZ*DT
      CDTH = 0.5D0*CDT
c
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C==== CEPS and CEPS21 are used in the kinetic propagation ====
C====== CAX and CBX are used in the kinetic propagation ======
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
      CEPS = CZ*CDT*RY/2.D0/AHC/DX/DX
      CEPS21 = 1.D0 - 2.D0 * CEPS
      CAX = -CEPS
      CBX = 1.D0 + 2.D0 * CEPS
      CPOTCOE = -CZ*RY*CDTH/AHC
C
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C========= Here we calculate the external potential ==========
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c      CALL POTEXT(V0,X,ELFR,CPOTCOE,WB,XK,POT,CV)
      CALL POTEXT(V0,X,ELFR,CPOTCOE,WB,POT,CV)
C
      WRITE(16,308) (X(IX)*A0,POT(IX)*RY,IX=1,NX)
C
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
C
      WRITE(18,*) '**********************************************'
      WRITE(18,*) '**********************************************'
      WRITE(18,*) '                                              '
      WRITE(18,*) '      INITIAL TIME T0 =',T0
      WRITE(18,*) '                                              '
      WRITE(18,*) '**********************************************'
      IF(IMR.EQ.0) THEN
      WRITE(18,*) '######### IMAGINARY TIME PROPAGATION #########'
      ELSE
      WRITE(18,*) '#######***# REAL TIME PROPAGATION #***########'
      END IF
      WRITE(18,*) '**********************************************'
      WRITE(18,*) '                                              '
      WRITE(18,*) 'SIZE OF THE SYSTEM (ANGSTRON) = ', WX
      WRITE(18,*) 'DT (fsec) = ', DT,'  # OF DT = ',NT
      WRITE(18,*) 'A0 (ANGSTRON) = ',A0,' RY (meV) = ',RY
      WRITE(18,*) 'MASS = ',AM,' Q = ',1.D0 / EP0
      WRITE(18,*) 'NUMBER OF GRID POINTS = ', NX
      WRITE(18,*) 'EXT. ELECTRIC FIELD (kV/cm)=', ELF
      WRITE(18,*) 'MAGNETIC FIELD (TESLA)=', BF
      WRITE(18,*) 'Kx (Kx*A0) =', xk
C
      IF(IMR.EQ.0) THEN
C
       DO 8 IT = 1, NT
C
        TI = T0 + IT*DT
C
        CALL WAVES(CV,CPSI)
C
        IF(MOD(IT,1000).EQ.0) THEN
C
         CALL ENERG(CPSI,POT)
         IF(NORB.GT.1) CALL ORDEM(CPSI)
         WRITE(17,307) TI,( EV(I)*RY, I=1,NORB)
         WRITE(*,307) TI, ( EV(I)*RY, I=1,NORB)
C
        END IF
C
8      CONTINUE
c
       CALL XAVG(X,CPSI,XAV,XAV2)
C
      ELSE
C
          IF(IPROJ.EQ.0) THEN
C
        DO 880 IT = 1, NT
C
           DTI = IT * DT
C
           CALL PROJECTION(CPSI0,CPSI,CPT)
           WRITE(80,*) CPT
         CALL WAVESREAL(CV,CPSI)
           IF(MOD(IT,100).EQ.0) THEN
            WRITE(*,*) 'T =', DTI
          CALL ENERG(CPSI,POT)
          WRITE(17,307) DTI/1000, ( EV(I)*RY, I=1,NORB)
C         WRITE(*,307) TI, ( EV(I)*RY, I=1,NORB)
         END IF
C
880      CONTINUE
C
          ELSE
C                
         IF(IDYN.EQ.0) THEN
C
            DO 88 IT = 1, NT
C
             DTI = IT * DT
           TI = DTI + T0
             WT = WD * DTI
C        
             CALL POTTIME(V0,X,EFD,CPOTCOE,WB,WT,POT,POTT,CV)
C
           CALL WAVESREAL(CV,CPSI)
C
          IF(MOD(IT,1).EQ.0) THEN
           CALL ENERG(CPSI,POTT)
           WRITE(17,307) DTI/1000, (EV(I)*RY, I=1,NORB)
C          WRITE(*,307) TI, ( EV(I)*RY, I=1,NORB)
             CALL PROBABIL(CPSI0,CPSI,PROB)
           WRITE(59,307) DTI/1000, (PROB(I), I=1,NORB)
C             CALL DIPOLO(X,CPSI,DIP)
C          WRITE(52,307) DTI, ( DIP(I), I=1,NORB)
           CALL RATIO(CPSI,PR)
           WRITE(19,307) DTI/1000, (PR(I), I=1,NORB)
           CALL XAVG(X,CPSI,XAV,XAV2)
           WRITE(51,307) DTI/1000, (XAV(I)*A0,XAV2(I)*A0, I=1,NORB)
            END IF
          IF(MOD(IT,5000).EQ.0) THEN
             WRITE(*,*) 'T =', DTI
C             DO 8383 IX = 1, NX
C             WRITE(70,307) X(IX)*A0, DCONJG(CPSI(IX,1))*CPSI(IX,1)
            END IF
C
88       CONTINUE
C
          ELSE
C
             DO 8000 IT = 1, NT
C
             DTI = IT * DT
           TI = DTI + T0
             WT = WD * DTI
C        
           CALL WAVESREAL(CV,CPSI)
C
          IF(MOD(IT,10).EQ.0) THEN
           CALL ENERG(CPSI,POT)
           WRITE(17,307) DTI/1000, (EV(I)*RY, I=1,NORB)
C          WRITE(*,307) TI, ( EV(I)*RY, I=1,NORB)
             CALL PROBABIL(CPSI0,CPSI,PROB)
           WRITE(59,307) DTI/1000, (PROB(I), I=1,NORB)
C             CALL DIPOLO(X,CPSI,DIP)
C          WRITE(52,307) DTI, ( DIP(I), I=1,NORB)
           CALL XAVG(X,CPSI,XAV,XAV2)
           WRITE(51,307) DTI/1000, (XAV(I)*A0,XAV2(I)*A0, I=1,NORB)
            END IF
          IF(MOD(IT,5000).EQ.0) THEN
             WRITE(*,*) 'T =', DTI
C             DO 8383 IX = 1, NX
C             WRITE(70,307) X(IX)*A0, DCONJG(CPSI(IX,1))*CPSI(IX,1)
            END IF
C
8000     CONTINUE
        END IF
       END IF
      END IF
C
       T0 = TTOT
C
       WRITE(18,*) '                                              '
       WRITE(18,*) '                                              '
       WRITE(18,*) '**********************************************'
       WRITE(18,*) 'STATE  ENERGY(meV)  ENERGY(GHz)   Xav(A)    PR'
       WRITE(18,*) '**********************************************'
       WRITE(18,*) '                                              '
C
C>>>>>>>>>>>>> PARTICIPATION RATIO <<<<<<<<<<<<
C
       DO 8732 IO = 1, NORB
         SUMP = 0.D0
         DO 8731 IX = 1, NX
           AUX = DCONJG(CPSI(IX,IO)) * CPSI(IX,IO)
           SUMP = AUX * AUX + SUMP
8731     CONTINUE
         PR(IO) = SUMP*DX 
C         WRITE(19,*) EV(IO)*RY, PR(IO)
8732   CONTINUE
C
C>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
C
         GHZ = 1000.D0 * RY / 4.1357                ! meV -> GHz
         DO 8832 IO = 1, NORB
        WRITE(18,301) IO,EV(IO)*RY,EV(IO)*GHZ,XAV(IO)*A0,PR(IO)
        WRITE(19,310) EV(IO)*RY,XAV(IO)*A0,XAV2(IO)*A0,PR(IO)
8832         CONTINUE
C
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       NI = 1
       NF = NORB
C       NI = 1
C       NF = 5
C3554   IF(NF.GT.NORB) NF = NORB
C       IF(NI.GT.NORB) GO TO 3553
       DO 99 IX = 1, NX, 2
        DO 9 IO = NI, NF
         PSI(IO) = DCONJG(CPSI(IX,IO)) * CPSI(IX,IO)
         IF(PSI(IO).LE.1.D-20) PSI(IO) = 0.D0
9       CONTINUE
C        IF(DABS(X(IX)).LE.50.1D0/A0) THEN
C         WRITE(50,302) X(IX)*A0, ( PSI(IO)/A0, IO = NI, NF )
C        END IF
        WRITE(20,310) X(IX)*A0,( PSI(IO)/A0, IO = NI, NF )
99     CONTINUE
C       NI = NI + 5
C       NF = NF + 5
C       GO TO 3554
C3553   CONTINUE
C
C       CALL RATIO(CPSI,PR)
C         DO I = 1, NORB
C        WRITE(19,310) EV(I)*RY, PR(I), XAV(I)*A0, XAV2(I)*A0
C         END DO
C
       WRITE(12) T0
C       WRITE(12) CPSI/DSQRT(A0)
       WRITE(12) CPSI
       REWIND(12)
       CLOSE(12)
C     
C      WRITE(21,*) ELF, BF, ( EV(IO)*RY,IO=1, NORB ) 
C
      WRITE(18,*) '                                              '
      WRITE(18,*) '**********************************************'
      WRITE(18,*) '>>>>>>>>>> WAVEFUNCTIONS PROJECTION <<<<<<<<<<'
      WRITE(18,*) '**********************************************'
      WRITE(18,*) '                                              '
C
       DO 10 I = 1, NORB
       DO 10 J = 1, I
        CSUM = (0.D0,0.D0)
        DO 135 IX = 1, NX
         CSUM = CSUM + DCONJG(CPSI(IX,I)) * CPSI(IX,J)
135     CONTINUE
        CPROJ = DX * CSUM
        WRITE(18,302) I, J, REAL(CPROJ), DIMAG(CPROJ)
10     CONTINUE
C
6     CONTINUE
      STOP
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE WAVEI(X0R,CPSI)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      DOUBLE COMPLEX CPSI(NX,NORB)
      DIMENSION X0R(NX), HERM(NORB,NX)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DO 101 J = 1, NX
       HERM(1,J) = 1.D0
       IF(NORB.GT.1) HERM(2,J) = 2.D0 * X0R(J)
101   CONTINUE
      IF(NORB.LT.3) GO TO 102
      DO 103 IX = 1, NX
      DO 103 IO = 3, NORB
       HERM(IO,IX) = 2.D0*(X0R(IX)*HERM(IO-1,IX)-(IO-1)*HERM(IO-2,IX))
103   CONTINUE
102   DO 104 I = 1, NX
       AUX = 0.5D0*X0R(I)**2
       IF(AUX.GT.30.D0) AUX = 30.D0
      DO 104 J = 1, NORB
       IF(J.LE.15) THEN
         L = J 
       ELSE
         L = J - 15 
       END IF
       CPSI(I,J) = HERM(L,I) * DEXP( -AUX )
104   CONTINUE
C
C >>>>> Normalizing waves function <<<<<<
C
      DO 105 I = 1, NORB
       SUM = 0.D0
       DO 106 IX = 1, NX
        SUM = SUM + DCONJG(CPSI(IX,I))*CPSI(IX,I)
106    CONTINUE
       AUX = DX * SUM
       ANORM = 1.D0 / DSQRT(AUX)
       DO 107 IX = 1, NX
        CPSI(IX,I) = ANORM * CPSI(IX,I)
107    CONTINUE
105   CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE POTEXT(V0,X,ELFR,CPOTCOE,WB,POT,CV)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10,NP=2)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DOUBLE COMPLEX CV(NX)
      DIMENSION X(NX),POT(NX),POTB(NX),POTE(NX),XL(NP),XR(NP)
      ALXH = 0.5D0 * ALX 
      A = 70.D0/A0
      B = 10.D0/A0
      D = A + B
        XLL = 250.D0 / A0
      BL = XLL - ALXH
      DO 100 I = 1, NP
       XL(I) = BL + (I-1)*D
       XR(I) = XL(I) + A
100   CONTINUE
      BM = BL + NP*D - B
      DO 400 IX = 1, NX
       POTB(IX) = 0.25D0 * ( WB * X(IX) )**2
       POTE(IX) = -ELFR * X(IX)
       IF(X(IX).LE.BL) POTE(IX) = -ELFR * BL
       IF(X(IX).GE.BM) POTE(IX) = -ELFR * BM
       DO 500 I = 1, NP
        IF(X(IX).LE.XR(I).AND.X(IX).GE.XL(I)) THEN
          POTE(IX) = -V0 + POTE(IX)
        END IF
500    CONTINUE
       POT(IX) = POTE(IX) + POTB(IX)
       CAUX = CPOTCOE * POT(IX)
       CV(IX) = CDEXP( CAUX )
400   CONTINUE
      RETURN
      END
C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE POTEXTN(V0,X,EF,CPOTCOE,WB,XK,POT,CV)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DOUBLE COMPLEX CV(NX)
      DIMENSION X(NX),POT(NX),POTB(NX)
        BETA = 1.01D0 / A0
        Z0 = -0.5D0 * WB * XK
        DO 1 IX = 1, NX
       POTB(IX) = 0.25D0 * ( WB * ( X(IX) - Z0 ) )**2
         IF(X(IX).LT.0.D0) POT(IX) = V0 - EF * X(IX) + POTB(IX)
         IF(X(IX).GE.0.D0) POT(IX) = -2.D0/(X(IX)+BETA) - EF * X(IX)
        1                           + POTB(IX)
       CAUX = CPOTCOE * POT(IX)
       CV(IX) = CDEXP( CAUX )
1     CONTINUE
      RETURN
      END
C
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE POTEXTHe(V0,X,EF,CPOTCOE,WB,XK,POT,CV)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DOUBLE COMPLEX CV(NX)
      DIMENSION X(NX),POT(NX),POTB(NX)
        BETA = 1.01D0 / A0
        DELTA = 1.D0 / A0
        Z0 = -0.5D0 * WB * XK
        DO 1 IX = 1, NX
         AUX = X(IX)/DELTA
         IF(AUX.GE.120.D0) AUX = 120.D0
       POTB(IX) = 0.25D0 * ( WB * ( X(IX) - Z0 ) )**2
         VX = - EF * X(IX) + POTB(IX)
         POT(IX) = VX + V0/(DEXP(AUX) + 1.D0 )
         IF(X(IX).GT.0.D0) POT(IX) = POT(IX) -2.D0/X(IX)
       CAUX = CPOTCOE * POT(IX)
       CV(IX) = CDEXP( CAUX )
1     CONTINUE
      RETURN
      END
C
C==================================================================
C
      SUBROUTINE POTEXTo(V0,X,EF,CPOTCOE,WB,XK,POT,CV)
C
C==================================================================
C
      PARAMETER(NX=1690,NP=9)        ! NP MUST BE ODD
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      DIMENSION X(NX),POT(NX),XL(NP),XR(NP)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DOUBLE COMPLEX CV(NX)
      A = 100.E0 / A0        ! well
        AH = 0.5D0 * A
      B = 40.E0 / A0        ! barrier
      D = A + B        ! superlattice period
        ALXH = 0.5D0 * ALX
        V0E = V0
        NPH = NP / 2
        TAMSLH = 0.5D0 * ( NP * D - B )
      DO 100 I = 1, NP
       XL(I) = -TAMSLH + ( I - 1 ) * D
       XR(I) = XL(I) + A
100   CONTINUE
        DO 1 IX = 1, NX
         POT(IX) = EF * X(IX)
       DO 500 I = 1, NP
        IF(X(IX).LE.XR(I).AND.X(IX).GT.XL(I))POT(IX) = -V0E + EF * X(IX)
500    CONTINUE        
         IF(DABS(X(IX)).GE.TAMSLH)POT(IX) =EF * TAMSLH * X(IX)/DABS(X(IX))
       CAUX = CPOTCOE * POT(IX)
       CV(IX) = CDEXP( CAUX )
1        CONTINUE
      RETURN
      END
c
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE POTTIME(V0,X,EF,CPOTCOE,BF,WT,POT,POTT,CV)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DOUBLE COMPLEX CV(NX)
      DIMENSION X(NX),POT(NX),POTT(NX)
        DO 1 IX = 1, NX
c         POTT(IX) = -EF * X(IX) * DCOS(WT)
         POTT(IX) = -EF * X(IX) * DSIN(WT)
       CAUX = CPOTCOE * ( POT(IX) + POTT(IX) )
       CV(IX) = CDEXP( CAUX )
1     CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE WAVES(CV,CPSI)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      DOUBLE COMPLEX CPSI(NX,NORB),CFIX(NX),CW(NX),CV(NX),
     1        CAK(NX),CGRAM(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      COMMON/WVFCS/ CEPS, CEPS21, CAX, CBX
C
       DO 107 IO = 1, NORB
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>>>> Psi(r) = DEXP[ -iV(r) dt/2 ] * Psi(r) <<<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
       DO 108 IX = 1, NX
        CW(IX) = CPSI(IX,IO) * CV(IX)
108    CONTINUE
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>> Psi(K) = DEXP[ -i(K**2/2M) dt ] * Psi(K) <<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
       CFIX(1)  = CEPS*CW(2)    + CEPS21*CW(1)
       CFIX(NX) = CEPS*CW(NX-1) + CEPS21*CW(NX)
       DO 109 IX = 2, NX-1
        CFIX(IX) = CEPS*CW(IX-1)+CEPS21*CW(IX)
     1             + CEPS*CW(IX+1)
109    CONTINUE
       CALL TRIDAG(CAX,CBX,CFIX,CW)
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>>>> Psi(r) = DEXP[ -iV(r) dt/2 ] * Psi(r) <<<<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
       DO 112 IX = 1, NX
        CPSI(IX,IO) = CW(IX) * CV(IX)
112    CONTINUE
C
107    CONTINUE
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>>>>>>>>>>>> Normalize the states <<<<<<<<<<<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      DO 203 IO = 1, NORB
       SUM = 0.D0
       DO 214 IX = 1, NX
        SUM = SUM + DCONJG(CPSI(IX,IO)) * CPSI(IX,IO)
214    CONTINUE
       ANOR1 = DX * SUM
       ACONS = 1.D0 / DSQRT(ANOR1)
       DO 215 IX = 1, NX
        CPSI(IX,IO) = ACONS * CPSI(IX,IO)
215    CONTINUE
203   CONTINUE
C
      IF(NORB.LE.1) RETURN
C
        CALL GRAM(CPSI)
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>>>>>>>>>>>>>GRAM-SCHMIDT METHOD <<<<<<<<<<<<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
C    | J > = | J > - SUM_i [ | i > * < i | J >
C
C      DO 113 J = 2, NORB
C       DO 114 I = 1, J - 1 
C        CSUM = (0.D0,0.D0)
C        DO 314 IX = 1, NX
C         CSUM = CSUM + DCONJG(CPSI(IX,I)) * CPSI(IX,J)
C314     CONTINUE
C        CGRAM(I) = -DX * CSUM
C114    CONTINUE
C       DO 201 IX = 1, NX
C        CAK(IX) = 0.D0
C        DO 201 K = 1, J-1
C         CAK(IX) = CAK(IX) + CGRAM(K) * CPSI(IX,K)
C201    CONTINUE
C       DO 202 IX = 1, NX
C        CPSI(IX,J) = CPSI(IX,J) + CAK(IX)
C202    CONTINUE
C113   CONTINUE
C      DO 204 J = 1, NORB
C       SUM = 0.D0
C       DO 304 IX = 1, NX                           
C        SUM = SUM + DCONJG(CPSI(IX,J)) * CPSI(IX,J)
C304    CONTINUE
C       ANORJ = DX * SUM
C       ACONS = 1.D0 / DSQRT(ANORJ)
C       DO 305 IX = 1, NX
C        CPSI(IX,J) = ACONS * CPSI(IX,J)
C305    CONTINUE
C204   CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE WAVESREAL(CV,CPSI)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      DOUBLE COMPLEX CPSI(NX,NORB),CFIX(NX),CW(NX),CV(NX),
     1        CAK(NX),CGRAM(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      COMMON/WVFCS/ CEPS, CEPS21, CAX, CBX
C
       DO 107 IO = 1, NORB
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>>>> Psi(r) = DEXP[ -iV(r) dt/2 ] * Psi(r) <<<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
       DO 108 IX = 1, NX
        CW(IX) = CPSI(IX,IO) * CV(IX)
108    CONTINUE
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>> Psi(K) = DEXP[ -i(K**2/2M) dt ] * Psi(K) <<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
       CFIX(1)  = CEPS*CW(2)    + CEPS21*CW(1)
       CFIX(NX) = CEPS*CW(NX-1) + CEPS21*CW(NX)
       DO 109 IX = 2, NX-1
        CFIX(IX) = CEPS*CW(IX-1)+CEPS21*CW(IX)
     1             + CEPS*CW(IX+1)
109    CONTINUE
       CALL TRIDAG(CAX,CBX,CFIX,CW)
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C>>>>> Psi(r) = DEXP[ -iV(r) dt/2 ] * Psi(r) <<<<<
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
       DO 112 IX = 1, NX
        CPSI(IX,IO) = CW(IX) * CV(IX)
112    CONTINUE
C
107    CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE GRAM(CPSI)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
C>>>>>>>>>> MODIFIED GRAM-SCHMIDT METHOD <<<<<<<<<<<
C===================================================
C    | J > = | J > - SUM_i [ | i > * < i | J >
C
      PARAMETER(NX=1690,NORB=10)
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      DOUBLE COMPLEX CPSI(NX,NORB),CGRAM(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DO 113 IO = 1, NORB
       CALL NORM(NX,DX,CPSI(1,IO))
       DO 114 I = IO + 1, NORB 
        CPROD = 0.5D0 * (DCONJG(CPSI(1,IO))*CPSI(1,I)+
     1                  DCONJG(CPSI(NX,IO))*CPSI(NX,I) )
        DO 115 IX = 2, NX-1
         CPROD=CPROD+DCONJG(CPSI(IX,IO))*CPSI(IX,I)
115     CONTINUE
        CGRAM(I) = -DX * CPROD
114    CONTINUE
       DO 201 IX = 1, NX
        DO 201 K = IO + 1, NORB
         CPSI(IX,K)=CPSI(IX,K)+CGRAM(K)*CPSI(IX,IO)
201    CONTINUE
113   CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE NORM(NX,DX,CW)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      DOUBLE COMPLEX CW(NX)
      SUM = 0.5D0*(CW(1) * DCONJG(CW(1)) + CW(NX) * DCONJG(CW(NX)))
      DO 1 IX = 2, NX - 1
       SUM = SUM + CW(IX) * DCONJG(CW(IX))
1     CONTINUE
      ANORM = 1.D0 / DSQRT( SUM * DX )
      DO 2 IX = 1, NX
       CW(IX) = ANORM * CW(IX)
2     CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE TRIDAG(CA,CB,CR,CU)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT DOUBLE COMPLEX (C)
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      PARAMETER(NX=1690)
      DOUBLE COMPLEX CGAM(NX),CR(NX),CU(NX)
      CBET = CB
      CU(1) = CR(1)/CBET
      DO 110 J = 2, NX
       CGAM(J) = CA/CBET
       CBET = CB - CA*CGAM(J)
       CU(J) = (CR(J) - CA*CU(J-1))/CBET
110   CONTINUE
      DO 120 J = NX-1,1,-1
       CU(J) = CU(J) - CGAM(J+1)*CU(J+1)
120   CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE XAVG(X,CPSI,XAV,XAV2)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      DOUBLE COMPLEX CPSI(NX,NORB)
      DIMENSION X(NX), XAV(NORB), XAV2(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI
      DO 40 J = 1, NORB
       SUM = 0.D0
         SUM2 = 0.D0
       DO 41 I = 1, NX
        SUM = SUM + DCONJG(CPSI(I,J)) * X(I) * CPSI(I,J)
        SUM2 = SUM2 + DCONJG(CPSI(I,J)) * X(I)*X(I) * CPSI(I,J)
41     CONTINUE
       XAV(J) = SUM * DX
       XAV2(J) = DSQRT( SUM2*DX - XAV(J)*XAV(J) )
40    CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE ENERG(CPSI,POT)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      COMMON/FUNC/ EV(NORB)
      DOUBLE COMPLEX CPSI(NX,NORB), CDPS(NX,NORB)
      DIMENSION DENS(NX), POT(NX)
C
      DO 115 IO = 1, NORB
       CDPS(1,IO)  = CPSI(2,IO) - 2.0*CPSI(1,IO)
       CDPS(NX,IO) = CPSI(NX-1,IO) - 2.0*CPSI(NX,IO)
       DO 1110 IX = 2, NX - 1
        CDPS(IX,IO) = CPSI(IX+1,IO)+CPSI(IX-1,IO)-2.0*CPSI(IX,IO)
1110   CONTINUE
115   CONTINUE
C
        DO 2220 IO = 1, NORB
         SUM = 0.D0
         DO 2221 IX = 1, NX
          SUM = SUM + DCONJG(CPSI(IX,IO)) * CDPS(IX,IO) 
2221     CONTINUE
         EV(IO) = -SUM / DX
2220    CONTINUE
C
       DO 117 IO = 1, NORB
        SUM = 0.D0
        DO 116 IX = 1, NX
         SUM = SUM + DCONJG(CPSI(IX,IO)) * POT(IX) * CPSI(IX,IO)
116    CONTINUE
        EV(IO) = EV(IO) + SUM * DX
117    CONTINUE
       RETURN
       END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE ORDEM(V)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8)(A-H,O-Z)
      PARAMETER(NX=1690,NORB=10)
      COMMON/FUNC/ EV(NORB)
      DOUBLE COMPLEX V(NX,NORB),C
      DO 13 I=1,NORB-1
       K=I
       P=EV(I)
       DO 11 J=I+1,NORB
        IF(EV(J).LE.P) THEN
         K=J
         P=EV(J)
        END IF
11     CONTINUE
       IF(K.NE.I) THEN
        EV(K)=EV(I)
        EV(I)=P
         DO 12 IX=1,NX
          C=V(IX,I)
          V(IX,I)=V(IX,K)
          V(IX,K)=C
12       CONTINUE
       END IF
13    CONTINUE
      RETURN
      END
C
C==================================================================
C
      SUBROUTINE DIPOLO(X,CPSI,DIP)
C
C==================================================================
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      DOUBLE COMPLEX CPSI(NX,NORB)
        DIMENSION X(NX),DIP(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
        DO 1 IO = 1,NORB
         SUM = 0.E0
         DO 2 IX = 1, NX
          SUM = SUM + CONJG(CPSI(IX,IO))*X(IX)*CPSI(IX,IO)
2         CONTINUE
         DIP(IO) = SUM * DX
1        CONTINUE
        RETURN
        END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE PROBABIL(CPSI0,CPSI,PROB)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CC
      PARAMETER(NX=1690,NORB=10)
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      DOUBLE COMPLEX CPSI0(NX,NORB),CPSI(NX,NORB)
        DIMENSION PROB(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
        DX2 = DX * DX
      DO 113 IO = 1, NORB
         CSUM = 0.D0
       DO 201 IX = 1, NX
          CSUM = CSUM + DCONJG(CPSI0(IX,IO)) * CPSI(IX,IO)
201    CONTINUE
c       PROB (IO) = 1.D0 - DCONJG(CSUM) * CSUM * DX2
       PROB (IO) = DCONJG(CSUM) * CSUM * DX2
113   CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE PROJECTION(CPSI0,CPSI,CPT)
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      PARAMETER(NX=1690,NORB=10)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
C      DOUBLE COMPLEX CPSI0(NX,NORB),CPSI(NX,NORB),CPT(NORB)
      DOUBLE COMPLEX CPSI0(NX,NORB),CPSI(NX,NORB)
C
C      DO 117 IO = 1, NORB
         CSUMX = 0.D0
       DO 116 IX = 1, NX
C        CSUMX = CSUMX + DCONJG(CPSI0(IX,IO)) * CPSI(IX,IO) 
        CSUMX = CSUMX + DCONJG(CPSI0(IX,1)) * CPSI(IX,1) 
116    CONTINUE
C       CPT(IO) = CSUMX * DX 
       CPT = CSUMX * DX 
C117   CONTINUE
      RETURN
      END
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
C
      SUBROUTINE RATIO(CPSI,PR)
C>>>>>>>>>>>>> PARTICIPATION RATIO <<<<<<<<<<<<
C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
CC
      PARAMETER(NX=1690,NORB=10)
      IMPLICIT REAL(8) (A,B,D-H,O-Z)
      IMPLICIT DOUBLE COMPLEX (C)
      DOUBLE COMPLEX CPSI(NX,NORB)
        DIMENSION PR(NORB)
      COMMON/GERAL/ DX, DT, ALX, RY, PI, A0
      DO 1 IO = 1, NORB
       SUMP = 0.D0
       DO 2 IX = 1, NX
        AUX = DCONJG(CPSI(IX,IO)) * CPSI(IX,IO)
        SUMP = AUX * AUX + SUMP
2      CONTINUE
       PR(IO) = 1.D0 / SUMP / DX / ALX / A0 
1     CONTINUE
      RETURN
      END