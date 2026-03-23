%VTletter.cls -- A document class for writing business letters with VT letterhead. 
%This class adapted from "OSUletter.cls", copyright 2007, by Brian Wood (brian.wood@oregonstate.edu). 
%This version by Robert Browder and Carrie Cross, January/February 2018.
% Modifications to layout by LianTze Lim, 1/18/2018 and 2/12/2020. 
% Notes:
%
% 1. This class can be used for creating letters with VT logo at the
%    header and department address at the footer.
% 2. It is based on the standard LaTex document class, letter.cls.
% 3. Required packages are epsfig and ifthen.
% 4. If VT logo is to be included in the head, then the file
% HorizontalStacked_CMYK-2.pdf must be provided. Otherwise, the logo is omitted.
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{VTletter}
\RequirePackage{graphicx}
\RequirePackage{epsfig}
\RequirePackage{ifthen}
\RequirePackage{xcolor}
\DeclareGraphicsExtensions{.pdf}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{letter}}
\ProcessOptions
\LoadClass[letterpaper]{letter} % We always use letter paper.

\newcommand{\subjectname}{Subject}
\newcommand{\@subject}{}
\newcommand{\subject}[1]{\renewcommand{\@subject}{\subjectname: #1}}

\newcommand{\mnumbername}{Submission ID}
\newcommand{\@mnumber}{}
\newcommand{\mnumber}[1]{\renewcommand{\@mnumber}{\mnumbername: #1}}

\newboolean{logofound}
\IfFileExists{logo2.pdf}
    {\setboolean{logofound}{true}}
    {\setboolean{logofound}{false}}

\RequirePackage{geometry} % Sets layout lengths for whole document.
\geometry{
  letterpaper,
  textwidth=6.875in,
  textheight=8.5in,
  top=1in,
  footskip=.875in, % Moves footer up or down. Originally set to .75.
}

% The following adds vertical space when the letter starts so that the letter contents won't get too high on the page and overlap the logo and address.
\RequirePackage{etoolbox} 
\apptocmd{\letter}{\vspace*{1.5in}}{}{}

% textpos package positions the logo in an absolute position on the page.
\RequirePackage[absolute]{textpos}
\newlength{\addresswidth}
\setlength{\addresswidth}{2in}

% The following lines insert the logo and \newaddress when the letter starts.
\pretocmd{\letter}{%
  \begin{textblock*}{4in}(0.8125in,0.75in)
  \usebox{\departmenthead}
  \end{textblock*}

  \begin{textblock*}{\addresswidth}[1,0](7.6875in,0.75in)
  \small\newaddress
  \end{textblock*}
}{}{}

\usepackage[T1]{fontenc}
\usepackage{fontspec}

% The following disables ligatures in the footer text. We use this because other solutions (like the microtype package) are incompatible with XeLatex. 
%\newfontfamily\myfootfont[Scale=0.7, Ligatures={NoRequired,NoCommon,NoContextual}]{acherus_grotesque_-_bold.otf}
%\newfontfamily\footit[Scale=0.7, Ligatures={NoRequired,NoCommon,NoContextual}]{acherus_grotesque_-_regular_italic.otf}  

\newsavebox{\departmenthead}
\newsavebox{\departmentfoot}
\newsavebox{\emptyfoot}  
  
%HELP FOR SETTING UP PARBOXES
%\parbox[position][height][inner-pos]{width}{text}
% The \parbox has two mandatory arguments:
%     width - specifies the width of the parbox, and
%     text - the text that goes inside the parbox. 
% LaTeX will position a parbox so its center lines up with the center of the text line. The optional position argument allows you to line up either the top or bottom line in the parbox (default is top).
% If the height argument is not given, the box will have the natural height of the text.
% The inner-pos argument controls the placement of the text inside the box. If it is not specified, position is used.
%     t -- text is placed at the top of the box.
%     c -- text is centred in the box.
%     b -- text is placed at the bottom of the box.

\sbox{\departmenthead}{%
    \parbox[t]{4.0in}{\raggedright
        \ifthenelse{\boolean{logofound}}%
           %{\includegraphics[height=0.35in]{vtlogo.pdf}}
           % {\includegraphics[scale=1.5]{logo2.pdf}}
           {\includegraphics[scale=0.2]{logo_ifce.png}}
           {\parbox[t][1.0in][b]{2.0in}{\hfill} 
            \ClassWarning{VTletter}{HorizontalStacked_CMYK-2.pdf COULD NOT BE FOUND!}}
    }% 
}

\savebox{\emptyfoot}[\textwidth][c]{
    \hspace*{\textwidth}
}

\renewcommand{\ps@firstpage}{
    \renewcommand{\@oddhead}{}
    \renewcommand{\@oddfoot}{\usebox{\departmentfoot}}
    \renewcommand{\@evenhead}{}
    \renewcommand{\@evenfoot}{}
}

\renewcommand{\ps@empty}{
    \renewcommand{\@oddhead}{}
    \renewcommand{\@oddfoot}{\usebox{\departmentfoot}}
    \renewcommand{\@evenhead}{}
    \renewcommand{\@evenfoot}{\usebox{\departmentfoot}}
}

\providecommand{\@evenhead}{}
\providecommand{\@oddhead}{}
\providecommand{\@evenfoot}{}
\providecommand{\@oddfoot}{}

\pagestyle{empty} 

% We add the subject command to the opening.
\renewcommand{\opening}[1]{\thispagestyle{firstpage}%
    \ifx\@empty\fromaddress
    \else  % home address
        {\raggedleft
            \begin{tabular}{l@{}}\ignorespaces
            \fromaddress \\ *[1\parskip]%
       		\end{tabular}\par
        }%
     \fi
     %\vspace{-6\parskip}
     \vspace{-12\parskip}
     %\@date \vspace{2\parskip}\\
     \@date \vspace{1\parskip}\\
     {\raggedright \toname \\ \toaddress \par}%
     \vspace{1\parskip}%
     %\ifthenelse{\equal{\@subject}{}}{}{\@subject\par}
     %\ifthenelse{\equal{\@mnumber}{}}{}{\@mnumber\par}
     \@subject \\
     \@mnumber \\ \\
     \vspace{1\parskip}%
     #1\par\nobreak
}

% We push closing to the right.
\renewcommand{\closing}[1]{\par\nobreak\vspace{\parskip}%
    \stopbreaks
    \noindent
    \hspace*{0.6\textwidth}\parbox{0.4\textwidth}{\raggedright
    \ignorespaces #1\\[4\medskipamount]%
    \ifx\@empty\fromsig
        \fromname
    \else \fromsig 
    \fi\strut}%
    \par
}


