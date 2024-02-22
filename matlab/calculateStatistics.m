function calculateStatistics(dataPath,outputPath)

    % Explore main path and list available networks
    networkList=dir(dataPath);
    networkList=networkList(3:end);
    metrics={'mediaIntensidadPixeles','varianza'};%'mediaIntensidadPixeles','ssim','varianza'};

    % Iterate over networks (only directories)
    for i=1:length(networkList)
        if networkList(i).isdir
            networkName=networkList(i).name;
            networkPath=fullfile(dataPath,networkName);
            fprintf(['Processing network: ' networkName]);
            % Iterate over metrics
            for j=1:length(metrics)
                metricName=metrics{j};
                metricFilePath=fullfile(networkPath,[networkName,'_',metricName,'.csv']);
                disp(['Processing metric: ' metricName]);
                % Read data
                tableData=readtable(metricFilePath,'ReadRowNames',true,'Delimiter',...
                    ';','DecimalSeparator',',');
                nVars=size(tableData,2);
                % Define pairs of combinations
                datasetNames=tableData.Properties.RowNames;
                combinations=[1,2;1,3;2,3];
                combinationsNames=cell(size(combinations,1),1);
                for k=1:size(combinations,1)
                    combinationsNames{k}=[datasetNames{combinations(k,1)},'_vs_',datasetNames{combinations(k,2)}];
                end
                % Empy table to store ttest results
                ttestTable=table('Size',[size(combinations,1),2],'VariableTypes',{'double','double'},...
                    'VariableNames',{'ttestT','ttestP'},'RowNames',combinationsNames);
                % Iterate over combinations
                for c=1:size(combinations,1)
                    disp(['Processing combination: ',combinationsNames{c}]);
                    % Plot with legend and no graphics
                    figure('Visible','off');
                    scatter(1:nVars,tableData{combinations(c,1),:},'blue');
                    hold on;
                    scatter(1:nVars,tableData{combinations(c,2),:},'red');
                    legend(datasetNames{combinations(c,1)},datasetNames{combinations(c,2)});
                    % Save plot
                    plotPath=fullfile(outputPath,[networkName,'_',metricName,'_',...
                        combinationsNames{c},'.png']);
                    saveas(gcf,plotPath);
                    % Ttest
                    [ttestT,ttestP]=ttest2(tableData{combinations(c,1),:},tableData{combinations(c,2),:});
                    % Store ttest results in the table
                    ttestTable{combinationsNames{c},'ttestT'}=ttestT;
                    ttestTable{combinationsNames{c},'ttestP'}=ttestP;
                end
                % Write ttest results to file
                ttestPath=fullfile(outputPath,[networkName,'_',metricName,'_ttestResults.csv']);
                writetable(ttestTable,ttestPath,'WriteRowNames',true,'Delimiter',',');
            end
        end
    end

end