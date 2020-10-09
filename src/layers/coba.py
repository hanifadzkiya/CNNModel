

def backward(self, dy):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:, w:w+self.K, h:h+self.K]
                    dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

def backward(self, dl_dout, lrate, momentum_rate):
    dl_din = np.zeros(self.inputs_size)
    
    for k in range(dl_dout[2]):
        for j in range(dl_dout[1]):
            for i in range(dl_dout[0]):
                dl_df[:,k,:,:] += dl_dout[i,j,k] * self.input[]

def backward(self, dl_dout=0, lrate=0.01, momentum_rate=0.1):
        
        dl_din = np.zeros(self.inputs_size)              # loss gradient of the input to the convolution operation

        idx_x = 0
        outx = 0
        print(self.filter_size)
        print(dl_dout.shape)
        while idx_x + self.filter_size[0] <= self.inputs_size[0]:
            idx_y = 0
            outy = 0
            while idx_y + self.filter_size[1] <= self.inputs_size[1]:
                for k in range(self.n_filter):
                    mtrx_area = self.input[idx_x:idx_x + self.filter_size[0], idx_y:idx_y + self.filter_size[1], :]
                    print("mtrx", mtrx_area.shape)
                    print("dldoutss",dl_dout.shape)
                    print(mtrx_area)
                    print(np.sum(dl_dout[outx, outy, :] * mtrx_area, axis=2).shape)
                    
                    self.dl_df[:,k] += np.sum(dl_dout[outx, outy, :] * mtrx_area, axis=2)
                    dl_din[idx_x:idx_x + self.filter_size[0], idx_y:idx_y + self.filter_size[1], :] += dl_dout[outx, outy, :] * self.filter[:,k]
                    
                idx_y += self.n_stride
                outy += 1
            idx_x += self.n_stride
            outx += 1

        return dl_din  