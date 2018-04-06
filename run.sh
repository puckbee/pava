#! /bin/bash



function getDir()
{
#    x="a,b,c,d"
    OLD_IFS="$IFS"
    IFS="/"
    array=($1)
    IFS="$OLD_IFS"

    count=${#array[@]}
    echo ${array[count-2]}
}

function getFile()
{
#    x="a,b,c,d"
    OLD_IFS="$IFS"
    IFS="/"
    array=($1)
    IFS="$OLD_IFS"

    count=${#array[@]}
    echo ${array[count-1]}
}
prefix=../dataset/TAMU/

#dirList=(ACUSIM AG-Monien Alemdar AMD Andrews Andrianov ANSYS Arenas ATandT Averous Bai Barabasi Bates Belcastro BenElechi Bindel Bodendiek Boeing Bomhof Botonakis Bourchtein Bova Brethour Brogan Brunetiere Buss Bydder Cannizzo Castrillon CEMW Chen Chevron Clark Cote CPM Cunningham Cylshell Dattorro Davis Dehghani DIMACS10 DNVS DRIVCAV Dziekonski Engwirda FEMLAB FIDAP Fluorem FreeFieldTechnologies Freescale Gaertner Garon GHS_indef GHS_psdef Gleich Goodwin Graham Grund Gset Gupta Hamm Hamrle Harvard_Seismology HB Hohn Hollinger HVDC IBM_Austin IBM_EDA INPRO IPSO Janna JGD_BIBD JGD_CAG JGD_Forest JGD_Franz JGD_G5 JGD_GL6 JGD_GL7d JGD_Groebner JGD_Homology JGD_Kocay JGD_Margulies JGD_Relat JGD_SL6 JGD_SPG JGD_Taha JGD_Trefethen Kamvar Kemelmacher Kim Koutsovasilis Langemyr LAW Lee LeGresley Li Lin LiuWenzhuo Lourakis LPnetlib Lucifora Luong Mallya Mancktelow Marini MathWorks MaxPlanck Mazaheri McRae Meng Meszaros Mittelmann MKS Moqri Morandini Muite Mulvey Nasa ND Nemeth Newman Norris NYPA Oberwolfach Okunbor Pajek PARSEC Pereyra POLYFLOW Pothen Priebel Puri Qaplib QCD QLi Quaglino QY Rajat Raju Rommes Ronis Rost Rothberg Rucci Rudnyi Sandia Sanghavi Schenk Schenk_AFE Schenk_IBMNA Schenk_IBMSDS Schenk_ISEI Schmid Schulthess Shen Shyy Simon Sinclair SNAP Sorensen Springer Stevenson Sumner Szczerba TKK TOKAMAK Toledo Tromble TSOPF Um UTEP vanHeukelum VanVelzen VDOL Vavasis Wang Watson Williams Wissgott YCheng Yoshiyasu YZhou Zaoui Zhao Zitney)
dirList=(Schenk Schenk_AFE Schenk_IBMNA Schenk_IBMSDS Schenk_ISEI Schmid Schulthess Shen Shyy Simon Sinclair SNAP Sorensen Springer Stevenson Sumner Szczerba TKK TOKAMAK Toledo Tromble TSOPF Um UTEP vanHeukelum VanVelzen VDOL Vavasis Wang Watson Williams Wissgott YCheng Yoshiyasu YZhou Zaoui Zhao Zitney)

dirList2=(DIMACS10)

threads_array=(68 136 204 272)

startIndex=50
endIndex=50
index=0

if [ $# -ne 0 ]
then

    if [ $# -lt 3 ]
    then
    
        dir=`getDir $1`
        file=`getFile $1`

        echo " Now Benchmarking the File ${file}" | tee log/${dir}/log_${file}.txt

        numactl --membind=1 ./pava$2 ${prefix}${dir}/${file} 0 | tee log/${dir}/log_${file}.txt
#            echo RunningShell ${dir} $? | tee -a log/log_${dir}.txt
        if 
            grep -R NormalEnding log/${dir}/log_${file}.txt > /dev/null
            then
            echo Logging ${dir} ${file} Succeeded. | tee -a log/log_exit.txt
            echo
            echo
            else
            echo Logging ${dir} ${file} Failed. | tee -a log/log_exit.txt
            echo 
            echo
        fi
    fi

    if [ $# -eq 3 ]
    then
    
        dir=`getDir $1`
        file=`getFile $1`

        echo " Now Benchmarking the File ${file}" | tee log/${dir}/log_${file}.txt

        for threads in ${threads_array[@]}
        do
            numactl --membind=1 ./pava ${prefix}${dir}/${file} ${threads} | tee -a log/${dir}/log_${file}.txt
#            echo RunningShell ${dir} $? | tee -a log/log_${dir}.txt
        done

        if 
            grep -R NormalEnding log/${dir}/log_${file}.txt > /dev/null
            then
            echo Logging ${dir} ${file} Succeeded. | tee -a log/log_exit.txt
            echo
            echo
            else
            echo Logging ${dir} ${file} Failed. | tee -a log/log_exit.txt
            echo 
            echo
        fi
    fi
else

    for dir in ${dirList[@]}  
    do
        if [ $index -ge $startIndex -a $index -lt $endIndex ]
        then
            echo "==========================================="
            echo " Now we benchmarking the directory:  ${dir}"
            echo "==========================================="
            mkdir log/${dir}
            for file in `ls ${prefix}${dir}`
            do
                echo " Now Benchmarking the File ${file}" | tee log/${dir}/log_${file}.txt

                numactl --membind=1 ./pava ${prefix}${dir}/${file} | tee log/${dir}/log_${file}.txt
#            echo RunningShell ${dir} $? | tee -a log/log_${dir}.txt
                if 
                    grep -R NormalEnding log/${dir}/log_${file}.txt > /dev/null
                    then
                    echo Logging ${dir} ${file} Succeeded. | tee -a log/log_exit.txt
                    echo
                    echo
                    else
                    echo Logging ${dir} ${file} Failed. | tee -a log/log_exit.txt
                    echo 
                    echo
                fi
            done

#        res= 'ls'
#        echo res

#        numactl --membind=1 ./pava ${prefix}${dir} | tee log/log_${dir}.txt

#        numactl --membind=1 ./pava ${prefix}${dir}
        fi

        let "index++"
    done
fi

