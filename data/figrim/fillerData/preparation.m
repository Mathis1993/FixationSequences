%% Load and copy
load("allImages_fillers.mat")

all = allImages;

%% Delete empty cells
%userdata ist 1x68, weil insgesamt 68 verschiedene Probanden, und 
%jeder Proband hat eine feste Zeile; wenn also Bild 6 von Probanden 1 3 und
%5 gesehen wurde, sind diese Zeilen gefüllt und die anderen 65 sind leer.
%Stimmt diese Interpretation?
%Die leeren Zeilen werden gelöscht.

for i = 1:length(all)
    %iterate backwards to avoid index problems
    for j = length(all(i).userdata):-1:1
        if isempty(all(i).userdata(j).fixations)
            all(i).userdata(j) = [];
        end
    end
end

%% Simplify struct stored under "fixation" to matrix
%In der Cell "fixation" in "userdata" ist eine struct gespeichert, die 
%Felder für die Enkodierung (1. Präsentation), sowie für die 2. und 3.
%Präsentation enthält. Da es sich hier um filler handelt, gab es nur eine
%einzige (erste) Präsentation. Also unter "fixation" nur die Daten der 
%Enkodierung speichern, ohne die beiden leeren Felder für 2. und 3. 
%Präsentation

for i = 1:length(all)
    for j = 1:length(all(i).userdata)
        all(i).userdata(j).fixations = all(i).userdata(j).fixations.enc;
    end
end

%Nun ist unter "fixations" die Sequenz der Fixationen bei der
%Enkodierung (1.Präsentation) zu finden: Eine Matrix bestehend aus der
%ersten Fixation mit x- und y-Koordinate, gefolgt von der zweiten usw.

%% Save
save("allImages_edited.mat", "all");