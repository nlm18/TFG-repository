function calculateStatistics(dataPath,outputPath)

    % Explore main path and list available networks
    networkList=dir(dataPath);
    networkList=networkList(3:end);
    metrics={'mediaIntensidadPixeles','varianza'};%'mediaIntensidadPixeles','ssim','varianza'};

    % Define some global parameters
    combinationPairs=[1,2;1,3;2,3];
    ttestTableFull=table();
    
    % Iterate over networks (only directories)
    for i=1:length(networkList)
        if networkList(i).isdir
            networkName=networkList(i).name;
            networkPath=fullfile(dataPath,networkName);
            fprintf(['Processing network: ',networkName,'\n']);
            % Iterate over metrics
            for j=1:length(metrics)
                metricName=metrics{j};
                metricFilePath=fullfile(networkPath,[networkName,'_',metricName,'.csv']);
                fprintf(['\t','Processing metric: ',metricName,'\n']);
                % Read data
                tableData=readtable(metricFilePath,'ReadRowNames',true,'Delimiter',...
                    ';','DecimalSeparator',',');
                nVars=size(tableData,2);
                % Define names of combinationPairs
                datasetNames=tableData.Properties.RowNames;
                combinationsNames=cell(size(combinationPairs,1),1);
                for k=1:size(combinationPairs,1)
                    combinationsNames{k}=[datasetNames{combinationPairs(k,1)},'_vs_',datasetNames{combinationPairs(k,2)}];
                end
                % Empy table to store ttest results
                ttestTable=table('Size',[size(combinationPairs,1),5],'VariableTypes',...
                    {'string','string','string','double','double'},...
                    'VariableNames',{'network','metric','combination','ttestT','ttestP'});
                % Iterate over combinationPairs
                for c=1:size(combinationPairs,1)
                    fprintf(['\t','\t','Processing combination: ',combinationsNames{c},'\n']);
                    % Plot with legend and no graphics
                    figure('Visible','off');
                    scatter(1:nVars,tableData{combinationPairs(c,1),:},'blue');
                    hold on;
                    scatter(1:nVars,tableData{combinationPairs(c,2),:},'red');
                    legend(datasetNames{combinationPairs(c,1)},datasetNames{combinationPairs(c,2)});
                    % Save plot
                    plotPath=fullfile(outputPath,[networkName,'_',metricName,'_',...
                        combinationsNames{c},'.png']);
                    saveas(gcf,plotPath);
                    % Ttest
                    [ttestT,ttestP]=ttest2(tableData{combinationPairs(c,1),:},tableData{combinationPairs(c,2),:});
                    % Store ttest results in the table
                    ttestTable{c,'network'}=string(networkName);
                    ttestTable{c,'metric'}=string(metricName);
                    ttestTable{c,'combination'}=string(combinationsNames{c});
                    ttestTable{c,'ttestT'}=ttestT;
                    ttestTable{c,'ttestP'}=ttestP;
                end
                ttestTableFull=[ttestTableFull;ttestTable];
            end
        end
    end
    
    % Write global ttest results to file
    ttestPathFull=fullfile(outputPath,'ttestResultsFull.csv');
    writetable(ttestTableFull,ttestPathFull,'Delimiter',',');
    
    fprintf('Finished processing all networks and metrics\n');

end