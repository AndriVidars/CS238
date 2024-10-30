"""
    @lru_cache(maxsize=None)
    def m(self, i, j):
        if j:
            idx = np.logical_and.reduce([
                self.value_index_array[parent_node, parent_node_val - 1] for parent_node, parent_node_val in j
            ])
        else:
            idx = np.ones(self.n_obs, dtype=bool)

        i_vals = self.x[:, i][idx]
        counts = np.bincount(i_vals, minlength=self.x_values_range[i] + 1)
        m_ijk_ = counts[1:]  # Skip the zero index
        m_ij0 = idx.sum()
        return m_ijk_, m_ij0
    
    
    @lru_cache(maxsize=None)
    def get_parent_instantiations(self, parents):        
        parent_data = self.x[:, parents]
        instantiatons = np.unique(parent_data, axis=0) # only get unique instantiations
        return [tuple(zip(parents, combination)) for combination in instantiatons]
    
    def bayesian_score_(self):
        p = 0
        for i in range(self.n):
            parents = tuple(sorted(self.G.predecessors(i)))
                        
            q = self.get_parent_instantiations(parents) if parents else [()] # case where node has no parents
            r = self.x_values_range[i]
            for j in q:
                m_ijk, m_ij0 = self.m(i, j)
                alpha_ij0 = r # uniform prior, each alpha_ijk has value 1
                p += (loggamma(alpha_ij0) - loggamma(alpha_ij0 + m_ij0))
                p += sum(loggamma(1 + m_ijk[k]) for k in range(r)) # shifted for 0-indexing. Uniform prior, denominator term eliminated
  
        return p
    """