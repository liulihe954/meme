import os
import math

class MEMEoops:
    """Abstract base class for learning motif using MEME oops model.

    A functional class defines some basic methods: raw input data processing, generalization of E step and M step,
    selecting the best candidate motif/starting point and several other helper functions.
    """
    def __init__(self,input_path,W,output_paths,model="OOPS"):
        """Constructor that takes in model parameters"""
        self.input_path = input_path
        self.W = W
        self.model = model
        self.best_candidate = None
        self.best_candidate_loglikeli = None
        self.init_pwm = None
        self.init_z = None
        self.starting_pwm = None
        self.cur_likelihood = None
        self.output_paths = output_paths
        
    ###################################
    ### Init: read seqs and init pwm ###
    ##################################
    def enum_candidate(self):
        """Given a list of sequences, enumerate all possible motif candidates by exhaustively going
        over each possible position in all sequences.
        """
        ##
        assert self.model.upper() == "OOPS","Please provide valid model name. Note OOPS is the only model available at this time point."
        ##
        seq_list = []
        candidate_list = []
        all_seq_len = []
        with open(os.path.join(self.input_path)) as f:
            lines = f.readlines()
            for line in lines:
                tmp_seq = line.strip()
                #
                assert self.W >= 2 and self.W <= len(tmp_seq), "W too long for some input sequences, potentially input sequences have different lengths"
                for c in tmp_seq:
                    assert c in "ACGT", "invalid character encountered in some sequence(s)"
                #
                seq_list.append(tmp_seq)
                for i in range(len(tmp_seq)- self.W + 1):
                    candidate_list.append(tmp_seq[i:i + self.W])
        # save seq_list and candidate_list and char count
        self.seq_list = seq_list
        self.candidate_list = list(set(candidate_list))
        #
        char_count = {"A":0,"C":0,"G":0,"T":0}
        for i in self.seq_list:
            for g in i:
                char_count[g]+=1                
        self.char_count = char_count
        
    def init_pnz(self,candidate_motif,row_name = "ACGT",pi=0.7):
        """Initialize a pwm for a given motif candidate.
        
        Args:
            candidate_motif: the pwm for a given motif candidate
            row_names: all possible characters in order, default "ACGT"
            pi: the initial probablity for the dominate character, default: 0.7
        Returns:
            save the initial pwm and initial z matrix as instance attributes
        """
        for c in candidate_motif:
            assert c in "ACGT"
        # pwm init
        pwm = {row_name[0]:[0] * (len(candidate_motif)+1),
               row_name[1]:[0] * (len(candidate_motif)+1),
               row_name[2]:[0] * (len(candidate_motif)+1),
               row_name[3]:[0] * (len(candidate_motif)+1)}
        pwm[row_name[0]][0] = pwm[row_name[1]][0] = pwm[row_name[2]][0] = pwm[row_name[3]][0] = 0.25
        for i in range(0,len(candidate_motif)):
            for j in row_name:
                if j == candidate_motif[i]:
                    pwm[j][i+1] = pi
                else:
                    pwm[j][i+1] = round((1-pi)/(len(row_name)-1),5)
        # z init
        z = []
        for k in range(len(self.seq_list)):
            z.append([0.25] * self.W)
            
        # save init_z and init_pwm
        self.init_pwm = pwm
        self.init_z = z
    
    ##############################
    ### E: reestimate z with p ###
    ##############################
    """ The E step - reestimate z matrix (t) using a input pwm (t-1)
        Args:
            pwm: the pwm at time (t-1)     
        Returns:
            save the reestimated z matrix (t) in the instance field
    """
    ## 
    def E(self,pwm):
        if self.init_pwm is None:
            print("Please run picking_starting_point() to pick a starting postion first.")
            pass
        else:
            z_new = []
            for i in range(len(self.seq_list)):
                tmp_seq = self.seq_list[i]
                tmp_z_row = [0] * (len(tmp_seq)- self.W + 1)
                for j in range(len(tmp_z_row)):
                    tmp_z_row[j] = self.seq_prob_helper(tmp_seq, pwm,j)
                z_new.append(self.z_normalize_helper(tmp_z_row))
            #
            self.cur_z = z_new
            
    #
    def z_normalize_helper(self,tmp_z_row):
        """A helper function to normalize a raw newly updated Z matrix by row (let the row sum up to 1).
        Args:
            tmp_z_row: the z matrix to normlaize  
        Returns:
            a normalized z matrix
        """
        total = 0
        for v in tmp_z_row:
            total+=v
        return [z/total for z in tmp_z_row]

    def seq_prob_helper(self, individual_seq, pwm, j):
        """ A helper function to calcualte the probablity of observing a given sequence with the motif (of length W)
            starting at position j.
        Args:
            individual_seq: the input sequence 
            pwm: the current pwm 
            j: the target starting position of the motif (of length W)
        Returns:
            a probablity
        """
        seq_prob = 1
        t = 0
        for p in range(len(individual_seq)):
            if p >= j and p <= j + self.W - 1:
                t+=1
                seq_prob = seq_prob * pwm[individual_seq[p]][t]
            else:   
                seq_prob = seq_prob * pwm[individual_seq[p]][0]
        return seq_prob
    
    ##############################
    ### M: reestimate p with z ###
    ##############################
    def M(self,z,sudo_count = 1, row_name = "ACGT"):
        """ The M step: using the current Z matrix to reestimate the pwm
        Args:
            z: the current z matrix
            sudo_count: sudo count to add
            row_name: all possible characters in order, default "ACGT"
        Returns:
            save the reestimated pwm to instance field
        """
        assert len(self.seq_list) == len(z)
        #
        new_pwm = {row_name[0]:[0] * (self.W + 1),
                   row_name[1]:[0] * (self.W + 1),
                   row_name[2]:[0] * (self.W + 1),
                   row_name[3]:[0] * (self.W + 1)}
        
        # accumulate observed z (allocate to each position)
        for seq_tmp_idx in range(len(self.seq_list)):
            seq_tmp = self.seq_list[seq_tmp_idx]
            for j in range(len(seq_tmp)- self.W + 1):
                motif_tmp = seq_tmp[j:(j + self.W)]
                for k in range(len(motif_tmp)):
                    tmp_char = motif_tmp[k]
                    #print("now fou : " + tmp_char)
                    new_pwm[tmp_char][k+1] += z[seq_tmp_idx][j]

        # calculate background
        for c in row_name:
            tmp = new_pwm[c]
            new_pwm[c][0] = self.char_count[c] - sum(new_pwm[c][1:])
            
        # normalize
        new_pwm_norm = self.normlize_pwd(new_pwm)
        
        #
        self.cur_pwm = new_pwm_norm
        self.cur_likelihood = self.obtain_likelihood()

    #
    def normlize_pwd(self, pwm,row_name = "ACGT"):
        """ A helper method to normalize a pwm by column
        Args:
            pwm: the input pwm to normalize
        Returns:
            A normllized pwm
        """
        assert len(pwm) == 4
        for i in range(self.W + 1):
            tmp_sum = 4 + pwm[row_name[0]][i] + pwm[row_name[1]][i] + pwm[row_name[2]][i] + pwm[row_name[3]][i]
            for c in row_name:
                pwm[c][i]  = round((1+pwm[c][i])/tmp_sum,5)
        return pwm
    
    ########################################################
    ### likelihold: calculate likelihold using updated P ###
    ########################################################
    def obtain_likelihood(self):
        """ Calculate the current log_e(probablity) of observing all the sequence using current pwm
        """
        #
        log_prob_accum = 0
        for i in range(len(self.seq_list)): # each seq
            tmp_seq = self.seq_list[i]
            tmp_prob = 0
            for j in range(len(tmp_seq)- self.W + 1):
                tmp_prob += self.seq_prob_helper(tmp_seq,self.cur_pwm,j)
            #
            log_prob_accum += math.log(tmp_prob/len(self.seq_list))
        #
        return log_prob_accum
    
    ##############################
    ### picking starting point: ###
    ##############################
    def picking_starting_point(self):
        """ A helper function to iterate over all possible motif candidates and select the one 
            with minimal log_e likelihood
        """
        if not self.best_candidate == None:
            print("Starting point already exists: thus passing")
            pass
        else:
            print("Check all candidates for best starting position")
            for candidate in self.candidate_list:
                self.init_pnz(candidate)
                self.E(self.init_pwm)
                self.M(self.cur_z)
                tmp_likelihood = self.obtain_likelihood()
                if self.best_candidate_loglikeli is None or tmp_likelihood > self.best_candidate_loglikeli:
                    self.best_candidate = candidate
                    self.best_candidate_loglikeli = tmp_likelihood  
                    self.cur_likelihood = tmp_likelihood  
                    self.starting_pwm = self.init_pwm
                    self.cur_pwm = self.init_pwm 
            #
            print("Best candidate finalized")
            
    ####################################
    ###    convert z and output files ###
    ####################################
    def convertZ_writeSubseq(self):
        """ A helper function to convert and output the Z matrix, as well as motif list, as required
        """
        assert len(self.seq_list) == len(self.cur_z)
        out_z = []
        out_subseq = []
        for i in range(len(self.cur_z)):
            tmp_best_pos = self.tie_break_helper(self.cur_z[i])
            out_z.append(tmp_best_pos)
            tmp_best_motif = self.seq_list[i][tmp_best_pos:tmp_best_pos + self.W]
            out_subseq.append(tmp_best_motif)
        return out_z,out_subseq
        
    def tie_break_helper(self,inputlist):
        """ A helper function to convert z matrix to 0/1, tie breaking rule is picking right-most position
        """
        max_val = max(inputlist)
        return len(inputlist) - inputlist[::-1].index(max_val) - 1
    
    def outputfile(self):
        
        # output PWM
        f = open(self.output_paths[0], "w")
        for k,v in self.cur_pwm.items():
            tmp_v_str = "\t".join([str(round(item,3)) for item in v])
            tmp_row = k + "\t" + tmp_v_str + "\n"
            f.write(tmp_row)
        f.close()
        print("The final PWM saved in:", self.output_paths[0])
        
        #
        out_z,out_subseq = self.convertZ_writeSubseq()
        
        # output positions
        f = open(self.output_paths[1], "w")
        f.write("\n".join(str(posi) for posi in out_z))
        f.close() 
        print("Predicted positions saved in:",self.output_paths[1])
        
        # output subsequences
        f = open(self.output_paths[2], "w")
        f.write("\n".join(str(subseq) for subseq in out_subseq))
        f.close()
        print("Corresponding subsequences saved in:",self.output_paths[2])
        