import glob,subprocess as sp
import midi_manipulation as mm

directory="./Nottingham/*.mid"
midilist = glob.glob(directory, recursive=True)

output_file= "Nottingham.txt"
out= open(output_file,"w")

for i in midilist[:1]: 
    print(i)
    noteval=mm.midiToNoteStateMatrix(i)
    mm.enc_int(noteval,out)
    print(noteval)
    

