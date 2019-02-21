# This is just a short script to generate clusters of points for debugging the expand_by_mcc.py script.

makeContigs = function(x, y, contigNameBase, nContigs) {

    df = data.frame(V1=x, V2=y);     d = dist(df, method='euclidean')
    h = hclust(d);     t = cutree(h, k=nContigs)

    df$ContigBase = NULL
    df$ContigName = NULL
    for(i in 1:nContigs) {
        cMask = grepl(i, t)
        df$ContigBase[cMask] = paste(contigNameBase, i, sep='_')
        df$ContigName[cMask] = paste(df$ContigBase[cMask], 1:length(df$ContigBase[cMask]), sep='|')
    }
    return(df)
}

set.seed(12345)
n1 = 500; x1 = rnorm(n1, 90, 5); y1 = rnorm(n1, -1, 1)
bin1 = makeContigs(x1, y1, 'ContigA', 90)
bin1$BinID = 'Bin1'

n2 = 300; x2 = rnorm(n2, 125, 7); y2 = rnorm(n2, 4, 0.5)
bin2 = makeContigs(x2, y2, 'ContigB', 40)
bin2$BinID = 'Bin2'

n3 = 450; x3 = rnorm(n3, 75, 10); y3 = rnorm(n3, -4, 1)
bin3 = makeContigs(x3, y3, 'ContigC', 100)
bin3$BinID = 'Bin3'

# Bind bin data.frames into a single frame
mockDF = Reduce(function(...) merge(..., all=TRUE), list(bin1, bin2, bin3))

# Save the table
write.table(mockDF, 'mock.table.txt', sep='\t', quote=F, row.names=F)

# Store the plot
png('mock.table.png', width=1000, height=1000)
plot(mockDF$V1, mockDF$V2, col=gsub('Bin', '', mockDF$BinID), pch=19, cex=1.5)
dev.off()