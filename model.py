import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepJIT(nn.Module):
    def __init__(self, args):
        super(DeepJIT, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num        

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim) # embedding layer, which is used to convert the input into a dense vector
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks]) # convolutional layers, which are used to extract features from the input

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim) # embedding layer, which is used to convert the input into a dense vector
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks]) # convolutional layers, which are used to extract features from the input
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks]) # convolutional layers, which are used to extract features from the input
        # there are 2 convolutional layers for commit code, one for each line and one for each file
        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob) # dropout layer, which is used to prevent overfitting
        self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class) # output layer
        self.sigmoid = nn.Sigmoid() # sigmoid activation function

    def forward_msg(self, x, convs):
        """
        This function applies a series of convolutional layers to the input tensor, followed by ReLU activation,
        max pooling, and concatenation of the output tensors.

        Parameters:
        x (torch.Tensor): The input tensor. It should have dimensions (N, W, D), where N is the batch size,
                          W is the length of the input, and D is the dimension of the input.
        convs (nn.ModuleList): A list of convolutional layers to apply to the input tensor.

        Returns:
        torch.Tensor: The output tensor after applying the convolutional layers, ReLU activation, max pooling,
                      and concatenation. The output tensor has dimensions (N, len(Ks)*Co), where len(Ks) is the
                      number of kernel sizes and Co is the number of output channels of the convolutional layers.
        """
        x = x.unsqueeze(1)  # (N, Ci, W, D) # N is the batch size, Ci is the input channel, W is the length of the input, D is the dimension of the input
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks) # Co is the output channel
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks) # max pooling
        x = torch.cat(x, 1) # (N, len(Ks)*Co)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        """
        This function applies a series of convolutional layers to each line and each file in the input tensor,
        followed by reshaping the tensor, and applying the forward_msg function.

        Parameters:
        x (torch.Tensor): The input tensor. It should have dimensions (N, F, W, D), where N is the batch size,
                          F is the number of files, W is the length of the input, and D is the dimension of the input.
        convs_line (nn.ModuleList): A list of convolutional layers to apply to each line in the input tensor.
        convs_hunks (nn.ModuleList): A list of convolutional layers to apply to each file in the input tensor.

        Returns:
        torch.Tensor: The output tensor after applying the convolutional layers, reshaping, and applying the forward_msg function.
                      The output tensor has dimensions (N, F, len(Ks)*Co), where len(Ks) is the number of kernel sizes and
                      Co is the number of output channels of the convolutional layers.
        """
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.args.num_filters * len(self.args.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward(self, msg, code):
        """
        This function applies the forward_msg and forward_code functions to the input tensors, concatenates the output tensors,
        applies dropout, and passes the result through two fully connected layers with ReLU and sigmoid activation functions.

        Parameters:
        msg (torch.Tensor): The input tensor for the message. It should have dimensions (N, W, D), where N is the batch size,
                            W is the length of the input, and D is the dimension of the input.
        code (torch.Tensor): The input tensor for the code. It should have dimensions (N, F, W, D), where N is the batch size,
                             F is the number of files, W is the length of the input, and D is the dimension of the input.

        Returns:
        torch.Tensor: The output tensor after applying the forward_msg and forward_code functions, concatenation, dropout,
                      and passing through the fully connected layers with ReLU and sigmoid activation functions.
                      The output tensor has dimensions (N, 1), where N is the batch size.
        """
        x_msg = self.embed_msg(msg)  # Embed the message
        x_msg = self.forward_msg(x_msg, self.convs_msg)  # Apply the forward_msg function to the embedded message

        x_code = self.embed_code(code)  # Embed the code
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)  # Apply the forward_code function to the embedded code

        x_commit = torch.cat((x_msg, x_code), 1)  # Concatenate the output tensors
        x_commit = self.dropout(x_commit)  # Apply dropout
        out = self.fc1(x_commit)  # Pass the result through the first fully connected layer
        out = F.relu(out)  # Apply ReLU activation function
        out = self.fc2(out)  # Pass the result through the second fully connected layer
        out = self.sigmoid(out).squeeze(1)  # Apply sigmoid activation function and squeeze the output tensor
        return out