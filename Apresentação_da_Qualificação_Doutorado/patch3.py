import re

with open("main.tex", "r") as f:
    text = f.read()

# Replace the frame Content
pattern = r"\\begin\{frame\}\{Descarregamento: Total vs.~Parcial --- Trade-offs\}.*?\\end\{frame\}"
replacement = r"""\\begin{frame}{Descarregamento: Total vs.~Parcial --- Trade-offs}
    \\vspace{0.05cm}
    \\begin{columns}[T]
        %% ---- COLUNA TOTAL ----
        \\begin{column}{0.49\\textwidth}
            \\centering
            \\begin{tikzpicture}[>=stealth, scale=0.75, transform shape, font=\\sffamily]
                \\tikzset{cnode/.style={draw, rounded corners=3pt, thick, fill=white,
                    minimum width=1.1cm, minimum height=0.55cm,
                    font=\\scriptsize\\bfseries, drop shadow}}

                % Cabeçalho
                \\node[fill=blue!80!black, text=white, font=\\scriptsize\\bfseries,
                      minimum width=5.2cm, minimum height=0.45cm, rounded corners=3pt]
                      at (2.5, 5.2) {TOTAL};

                % Granularidade Total
                \\node[text=blue!75!black, font=\\scriptsize\\bfseries] at (2.5, 4.78) {Granularidade Total};
                \\node[cnode, fill=blue!10, draw=blue!80!black] at (1.0, 4.20) {0};
                \\node[cnode, fill=blue!40, draw=blue!80!black] at (4.0, 4.20) {1};
                \\draw[->, thick, blue!80!black] (1.55, 4.20) -- (3.45, 4.20)
                      node[midway, below, font=\\scriptsize]{Local $\\to$ Borda};
                \\draw[dashed, gray!40] (0, 3.75) -- (5.0, 3.75);

                % Complexidade Total
                \\node[text=blue!75!black, font=\\scriptsize\\bfseries] at (2.5, 3.50) {Complexidade Total};
                \\node[cnode, fill=blue!10, draw=blue!80!black] at (2.5, 3.00) {DECISÃO};
                \\draw[->, thick, blue!80!black] (2.5, 2.73) -- (1.0, 2.25)
                      node[left, font=\\scriptsize]{Local};
                \\draw[->, thick, blue!80!black] (2.5, 2.73) -- (4.0, 2.25)
                      node[right, font=\\scriptsize]{Borda};
                \\draw[dashed, gray!40] (0, 1.90) -- (5.0, 1.90);

                % Overhead Total
                \\node[text=blue!75!black, font=\\scriptsize\\bfseries] at (2.5, 1.65) {Overhead Total};
                \\node[cnode, fill=blue!10, draw=blue!80!black, minimum width=2.4cm]
                      at (2.5, 1.12) {RESULTADO ÚNICO};
                \\node[text=green!55!black, font=\\scriptsize\\bfseries] at (2.5, 0.65)
                      {\\checkmark\\ PRONTO};
                \\draw[dashed, gray!40] (0, 0.35) -- (5.0, 0.35);

                % Tolerância a Falhas
                \\node[text=blue!75!black, font=\\scriptsize\\bfseries] at (2.5, 0.10) {Tolerância a Falhas};
                \\node[cnode, fill=blue!10, draw=blue!80!black, minimum width=2.4cm]
                      at (2.5, -0.42) {TAREFA TOTAL};
                \\draw[ultra thick, red] (1.75, -0.15) -- (3.25, -0.70);
                \\draw[ultra thick, red] (1.75, -0.70) -- (3.25, -0.15);
                \\node[text=red!80!black, font=\\scriptsize\\bfseries] at (2.5, -1.05) {PERDA TOTAL};
            \\end{tikzpicture}
        \\end{column}

        %% ---- COLUNA PARCIAL ----
        \\begin{column}{0.49\\textwidth}
            \\centering
            \\begin{tikzpicture}[>=stealth, scale=0.75, transform shape, font=\\sffamily]
                \\tikzset{snode/.style={draw, rounded corners=3pt, thick, fill=white,
                    minimum width=0.95cm, minimum height=0.55cm,
                    font=\\scriptsize\\bfseries, drop shadow}}

                % Cabeçalho
                \\node[fill=orange!80!black, text=white, font=\\scriptsize\\bfseries,
                      minimum width=5.2cm, minimum height=0.45cm, rounded corners=3pt]
                      at (2.5, 5.2) {PARCIAL};

                % Granularidade Parcial
                \\node[text=orange!75!black, font=\\scriptsize\\bfseries] at (2.5, 4.78) {Granularidade Parcial};
                \\draw[fill=gray!10, draw=gray!60, thick, rounded corners=2pt]
                      (0.4, 4.08) rectangle (4.6, 4.50);
                \\draw[fill=orange!50, draw=orange!80!black, thick, rounded corners=2pt]
                      (0.4, 4.08) rectangle (2.78, 4.50);
                \\node[font=\\scriptsize] at (1.59, 4.29) {60\\% Local};
                \\node[font=\\scriptsize] at (3.69, 4.29) {40\\% Borda};
                \\draw[dashed, gray!40] (0, 3.75) -- (5.0, 3.75);

                % Complexidade Parcial
                \\node[text=orange!75!black, font=\\scriptsize\\bfseries] at (2.5, 3.50) {Complexidade Parcial};
                \\node[snode, fill=orange!10, draw=orange!80!black] (fat) at (2.5, 3.00) {FATIAMENTO};
                \\node[snode] (e1) at (1.0, 2.38) {Sub 1};
                \\node[snode] (e2) at (2.5, 2.38) {Sub 2};
                \\node[snode] (e3) at (4.0, 2.38) {Sub 3};
                \\draw[->, thick, orange!80!black] (fat.south west) -- (e1.north);
                \\draw[->, thick, orange!80!black] (fat.south)      -- (e2.north);
                \\draw[->, thick, orange!80!black] (fat.south east) -- (e3.north);
                \\draw[<->, dashed, red!70!black, thick] (e1.east) -- (e2.west);
                \\draw[<->, dashed, red!70!black, thick] (e2.east) -- (e3.west);
                \\node[text=red!80!black, font=\\scriptsize\\bfseries, fill=white, inner sep=1pt] at (2.5, 1.90) {SINCRONIA};
                \\draw[dashed, gray!40] (0, 1.90) -- (1.5, 1.90);
                \\draw[dashed, gray!40] (3.5, 1.90) -- (5.0, 1.90);

                % Overhead Parcial
                \\node[text=orange!75!black, font=\\scriptsize\\bfseries] at (2.5, 1.55) {Overhead Parcial};
                \\node[snode, fill=orange!10, draw=orange!80!black] at (1.0, 1.05) {$R_1$};
                \\node[snode, fill=orange!10, draw=orange!80!black] at (2.5, 1.05) {$R_2$};
                \\node[snode, fill=orange!10, draw=orange!80!black] at (4.0, 1.05) {$R_3$};
                \\node[snode, fill=red!10, draw=red!70!black, minimum width=2.2cm]
                      (mg) at (2.5, 0.45) {MERGE};
                \\draw[->, thick, orange!80!black] (1.0, 0.77) -- (mg.north west);
                \\draw[->, thick, orange!80!black] (2.5, 0.77) -- (mg.north);
                \\draw[->, thick, orange!80!black] (4.0, 0.77) -- (mg.north east);
                \\node[text=red!80!black, font=\\scriptsize\\bfseries, fill=white, inner sep=1pt] at (2.5, -0.05)
                      {OVERHEAD DE SINCRONIA};
                \\draw[dashed, gray!40] (0, -0.05) -- (0.5, -0.05);
                \\draw[dashed, gray!40] (4.5, -0.05) -- (5.0, -0.05);
                \\draw[dashed, gray!40] (0, 0.35) -- (5.0, 0.35);

                % Tolerância a Falhas
                \\node[text=orange!75!black, font=\\scriptsize\\bfseries] at (2.5, -0.20) {Tolerância a Falhas};
                \\node[snode, fill=orange!10, draw=orange!80!black] at (1.0, -0.72) {$T_1$\\,\\checkmark};
                \\node[snode, fill=red!10,    draw=red!80!black]    at (2.5, -0.72) {$T_2$\\,\\texttimes};
                \\node[snode, fill=orange!10, draw=orange!80!black] at (4.0, -0.72) {$T_3$\\,\\checkmark};
                \\node[text=red!80!black, font=\\scriptsize\\bfseries] at (2.5, -1.35)
                      {REEXEC $T_2$};
            \\end{tikzpicture}
        \\end{column}
    \\end{columns}
\\end{frame}"""

new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)
with open("main.tex", "w") as f:
    f.write(new_text)

