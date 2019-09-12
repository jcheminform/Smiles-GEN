"""
Copyright 2019 Ruud van Deursen, Firmenich SA.
Copyright 2017 Peter Ertl, Novartis AG.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


"""
File SmilesGEN_utils_fixed.py defines utility functions
to prepare the SMILES datasets to train generative
models for SMILES strings as published in:
GEN: Highly Efficient SMILES Explorer Using Autodidactic Generative Examination Networks;
Ruud van Deursen, Peter Ertl, Igor V Tetko, Guillaume Godin; <link>
Please cite the above publication when using this layer.
"""

import numpy
from numpy import random

#######################################
# Section with a class defining utils #
#######################################
class DataUtils:
    """
    Class DataUtils defines a class for data preparation
    to convert SMILES String to a training set.
    """
    
    def __init__(self,replacements={"[nH]":"A","Cl":"L","Br":"R","/":"","\\":"",
                                       "[C@@H]":"C","[C@H]":"C","[C@@]":"C","[C@]":"C"},
                 okchars="CFLRIONSAcons123456789=#()\n",maxlen=42,step=3):
        """
        Constructor of ErtlLSTMUtils.
        Input:
        replacements -- Dictionary with replacements.
                        Default replacements:
                        [nH] => A, Cl => L, Br => R, / => '',\ => '',
                        [C@H] => C, [C@@H] => C, [C@@] >= C and [C@] => C.
        okchars      -- Allowed characters in the String. Defaults are
                        defines for SMILES.
                        Default: 'CFLRIONSAcons123456789=#()'
        """
        # Define the characters with fixed size
        self.chars = sorted(list(okchars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
        # Define the specified length
        self.maxlen = maxlen
        self.step = step
        
        # Store the characters and replacements
        self.okchars = okchars
        self.replacements = replacements
    
    def Replace(self,smi):
        """
        Method replaces some characters in a SMILES String
        and returns a new SMILES String if all used characters
        are qualified.
        Input:
        smi          -- SMILES.
        replacements -- Dictionary with replacements.
                        Default replacements:
                        [nH] => A, Cl => L, Br => R, / => '',\ => '',
                        [C@H] => C, [C@@H] => C, [C@@] >= C and [C@] => C.
        okchars      -- Allowed characters in the String. Defaults are
                        defines for SMILES.
                        Default: 'CFLRIONSAcons123456789=#()'
        Return:
        Modified SMILES with simplified characters.
        """
        # Make sure to strip and work with a clean 
        # SMILES String.
        smi = smi.strip().split("\t")[0]
        okchars,replacements = self.okchars,self.replacements
        for char in replacements.keys():
            if char in smi:
                smi = smi.replace(char,replacements[char])
        
        # Validate SMIs for valid characters
        if all(c in okchars for c in smi):
            return smi
        else:
            return ""
        
    def Prepare(self,smilist,clear=True,updateChars=True,augment=None,naug=0,shuffle=True):
        """
        Method prepares a list with SMILES String and returns
        a list with line-separated SMILES Strings for training.
        The SMILES will be subjected to characters replacement
        Input:
        smilist      -- List with multiple SMILES.
                        This can be a list of SMILES or a string
                        defining a filename
        clear        -- Flag to clear input (default = True).
        augment      -- Method to augment the entry for an identical
                        entry represented with a different string.
                        Method syntax: augment(text,naug), where
                        text is the input string and naug the number
                        of augmented entries (default is None, not executed).
        naug         -- Number of augmentations (default = 0, not executed).
        shuffle      -- Flag to shuffle data (default = True).
        Return:
        List with fine-tuned SMILES.
        """
        # Check the type of smilist - if string consider this
        # this variable a file with SMILES and read all lines
        if type(smilist) == str:
            with open(smilist) as f:
                data = [line.strip() for line in f.readlines()]
        else:
            data = smilist
        
        # Augment/randomize if instructed to do so
        if naug>0 and augment is not None:
            data = [aug for smi in data for aug in augment(smi,naug)]  

        # Shuffle the data
        if shuffle:
            random.shuffle(data)
            
        # Create the replacement and return as a single
        # string with a line separator between the words.
        keep = [self.Replace(smi) for smi in data]
        keep = list(filter(lambda x: len(x)>0 and len(x)<=self.maxlen,keep))
        self.text = "\n".join(keep)+"\n" # Add a breaker for the last line.
        
        # Clear input to save memory
        if clear:
            data.clear()

        # Cache the data and return the joined list
        self.data = data
        return self.text
    
    def Encode(self,text):
        """
        Method encodes a list of Strings to vectors for the LSTM.
        Input:
        text   -- Text to be translated.
        maxlen -- Maximum sentence length (default = 40).
                  This length will be set to the longest observed
                  length, if the longest observed length is lower
                  than maxlen.
        step   -- Step (default = 3).
        """
        # Update maxlen based on the longest observed word
        # This is important, otherwise no sentences will be presented
        # to the network.
        maxlen,step = self.maxlen,self.step
        maxword = numpy.max([len(x) for x in text.split("\n")])
        maxlen = numpy.min([maxword,maxlen])
        self.maxlen = maxlen
        
        # Translate the sentences to little pieces.
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])

        # Vectorize the sentences in boolean format
        X = numpy.zeros((len(sentences), maxlen, len(self.chars)), dtype=numpy.bool)
        y = numpy.zeros((len(sentences), len(self.chars)), dtype=numpy.bool)
        for i,sentence in enumerate(sentences):
            for t,char in enumerate(sentence):
                X[i,t,self.char_indices[char]] = True
            y[i,self.char_indices[next_chars[i]]] = True

        # Done: Return X,Y and maxlen
        return X,y,maxlen
        
    def NumChars(self):
        """
        Return:
        Number of cached characters.
        """
        if self.chars is None:
            return 0
        return len(self.chars)
    
    def MaxLen(self):
        """
        Return:
        Maximum observed length.
        """
        if self.maxlen is None:
            return 0
        return self.maxlen
    
    def Text(self):
        """
        Return:
        The text corpus.
        """
        return self.text
    
    def LenText(self):
        """
        Return:
        Length of the cached text.
        """
        return len(self.text)
    
######################
# Section with tests #
######################
import unittest
class ErtlLSTMUtilsTest(unittest.TestCase):
    
    def setUp(self):
        """
        Setup.
        """
        pass
    
    def test_Replace_All(self):
        """
        Method validates the replacement of SMILES
        using the default dictionary.
        """
        Utils = DataUtils()
        for smi_in,smi_out in [
            ("c1[nH]ccc1","c1Accc1"),
            ("FCF","FCF"),
            ("ClC(Cl)=C(Cl)Cl","LC(L)=C(L)L"),
            ("BrCC","RCC"),
            ("ICC","ICC"),
            ("C/C=C/C","CC=CC"),
            ("F[C@@H](Cl)Br","FC(L)R"),
            ("F[C@H](Cl)Br","FC(L)R"),
        ]:
            self.assertEqual(smi_out,Utils.Replace(smi_in))
            
    def test_Prepare_List(self):
        """
        Method validates the correct transformation of a list.
        """
        # Define a list with SMILES and define the expected output
        # Generate the String and compare to the expected output
        Utils = DataUtils()
        smiles_in = list(["ClCCl","C/C=C/C","CCCC","[nH]1cccc1"])
        expected = "\n".join(["LCL","CC=CC","CCCC","A1cccc1"])
        actual = Utils.Prepare(smiles_in)
        self.assertEqual(expected,actual)
            
#############################################
# Section with main method to run tests cmd #
#############################################
if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'],  verbosity=2, exit=False)
