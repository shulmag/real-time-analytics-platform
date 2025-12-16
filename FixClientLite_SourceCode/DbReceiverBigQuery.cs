
namespace FixClientLite
{
    using System;
    using System.Collections.Generic;
    using System.Configuration;
    using System.Data;
    using System.Data.SqlClient;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;

    using Newtonsoft.Json;

    using QuickFix;
    using QuickFix.Fields;
    using QuickFix.FIX44;
    
    using Message = QuickFix.Message;
    using System.Globalization;
    using Google.Cloud.BigQuery.V2;
    using Newtonsoft.Json;
    using Google.Apis.Auth.OAuth2;
    using System.IO;


    /// <summary>The wellington receiver.</summary>
    public class DbReceiverBigQuery : MessageCracker, ICompanyMessageCracker
    {
        private log4net.ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

        /// <summary>
        /// The connection string  key.
        /// </summary>
        private string connectionString;

        /// <summary>
        /// The connection string  key.
        /// </summary>
        private string connectionStringKey;

        public string ProjectId { get; set; }
        public string DatsetId { get; set; }
        public string TableId { get; set; }
        private BigQueryClient client; 


        public string ConnectionStringKey
        {
            get
            {
                return connectionStringKey;
            }
            set
            {
                this.connectionStringKey = value;                
            }
        }

        public void CrackMessage(Message message, SessionID sessionID)
        {
            //Console.WriteLine(message.ToString());
            logger.DebugFormat("FromApp...{0}", message.ToString());
            string msgType = message.Header.GetString(Tags.MsgType);

            if (msgType.Equals(MsgType.INDICATION_OF_INTEREST))
            {
                Trace o = MappingHelper.GetTrace((IndicationOfInterest)message);
                SaveToBigQueryTrace(o);
            }            

            else
            {
                logger.ErrorFormat("Unknown message. Unable to parse the message. Fix message type= {0}", msgType);
            }
        }

  

        private void SaveToBigQueryTrace(Trace o)
        {
        try
        {
            // Create a BigQuery client using your project ID.
            

            // Create a BigQueryInsertRow and populate it with your object's properties.
            var row = new BigQueryInsertRow
        {
            { "RawFeedId", o.RawFeedId },
            { "CreatedDate", o.CreatedDate },
            { "FinraCreatedDate", o.FinraCreatedDate },
            { "MbsMessageType", (int?)o.MbsMessageType },
            { "FinraTradeTypeId", (int?)o.FinraTradeTypeId },
            { "Exchange", o.Exchange },
            { "RDID", o.RDID },
            { "Cusip", o.Cusip },
            { "Quantity", o.Quantity },
            { "Price", o.Price },
            { "ExecutionDateTime", o.ExecutionDateTime },
            { "SettlementDate", o.SettlementDate },
            { "Factor", o.Factor },
            { "Bsym", o.Bsym },
            { "SubProductType", o.SubProductType },
            { "OriginalMessageSeqNumber", o.OriginalMessageSeqNumber },
            { "MessageSeqNumber", o.MessageSeqNumber },
            { "OriginalDisseminationDate", o.OriginalDisseminationDate },
            { "DisseminationDate", o.DisseminationDate },
            { "HighPrice", o.HighPrice },
            { "LowPrice", o.LowPrice },
            { "LastSalePrice", o.LastSalePrice },
            { "QuantityIndicator", o.QuantityIndicator },
            { "Remuneration", o.Remuneration },
            { "SpecialPriceIndicator", o.SpecialPriceIndicator },
            { "ReportingPartySide", (int?)o.ReportingPartySide },
            { "SaleCondition3", o.SaleCondition3 },
            { "SaleCondition4", o.SaleCondition4 },
            { "ChangeIndicator", o.ChangeIndicator },
            { "AsOfIndicator", o.AsOfIndicator },
            { "ReportingPartyType", o.ReportingPartyType },
            { "ContraPartyType", o.ContraPartyType },
            { "AtsIndicator", o.AtsIndicator },
            { "Symbol", o.Symbol },
            { "RawFixMessageAsXml", o.RawFixMessageAsXml },
            // Assuming DateInserted is a timestamp field
            { "DateInserted", DateTime.UtcNow }
        };

            // Insert the row into BigQuery.
            client.InsertRow(this.DatsetId, this.TableId, row);

                //string query = $@"
                //    SELECT *
                //    FROM `{this.ProjectId}.{this.DatsetId}.{this.TableId}`
                //    WHERE RawFeedId = @rawFeedId
                //    ORDER BY DateInserted DESC
                //    LIMIT 1";

                //// Create the query parameter.
                //var parameters = new[]
                //{
                //    new BigQueryParameter("rawFeedId", BigQueryDbType.Int64, o.RawFeedId)
                //};

                //// Execute the query.
                //BigQueryResults results = this.client.ExecuteQuery(query, parameters);

                //// Process the results.
                //foreach (var resultRow in results)
                //{
                //    // Example: Log or process the retrieved row. Adjust based on your needs.
                //    Console.WriteLine("Retrieved row: " + resultRow.ToString());
                //}

            }
        catch (Exception ex)
        {
            logger.ErrorFormat("Error on saving message to BigQuery with params {0}. The error is: {1}",
                JsonConvert.SerializeObject(o), ex.ToString());
        }
    }
   

        private string GetCreateTableSqlTraces()
        {
            return $@"CREATE TABLE '{this.ProjectId}.{this.DatsetId}.{this.TableId}' (
  RawFeedId INT64 NOT NULL,
  CreatedDate TIMESTAMP NOT NULL,
  FinraCreatedDate TIMESTAMP NOT NULL,
  MbsMessageType INT64 NOT NULL,
  FinraTradeTypeId INT64 NOT NULL,
  Exchange STRING,
  RDID STRING,
  Cusip STRING,
  Quantity INT64,
  Price FLOAT64,
  ExecutionDateTime TIMESTAMP,
  SettlementDate TIMESTAMP,
  Factor FLOAT64,
  Bsym STRING,
  SubProductType STRING,
  OriginalMessageSeqNumber INT64,
  MessageSeqNumber INT64,
  OriginalDisseminationDate TIMESTAMP,
  DisseminationDate TIMESTAMP,
  HighPrice FLOAT64,
  LowPrice FLOAT64,
  LastSalePrice FLOAT64,
  QuantityIndicator STRING,
  Remuneration STRING,
  SpecialPriceIndicator STRING,
  ReportingPartySide INT64,
  SaleCondition3 STRING,
  SaleCondition4 STRING,
  ChangeIndicator STRING,
  AsOfIndicator STRING,
  ReportingPartyType STRING,
  ContraPartyType STRING,
  AtsIndicator STRING,
  Symbol STRING,
  RawFixMessageAsXml STRING,
  DateInserted TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
);

";
        }

        public void RequestOfferings(SessionID sessionId)
        {
            throw new NotImplementedException();
        }

        public void OnLogon(SessionID sessionId)
        {

        }

        public void OnLogout(SessionID sessionId)
        {

        }

        public void SendMessage(Message fixMessage, SessionID sessionId)
        {
            throw new NotImplementedException();
        }
        public void Initialize(string aa)
        {
            throw new NotImplementedException();
        }

        public void Initialize(string credentialsKeyFilePath, string projectId, string datasetId, string tableId)
        {
            this.ProjectId = projectId;
            this.DatsetId = datasetId;
            this.TableId = tableId;
            GoogleCredential credential;
            using (var stream = new FileStream(credentialsKeyFilePath, FileMode.Open, FileAccess.Read))
            {
                credential = GoogleCredential.FromStream(stream);
            }

            // Create the BigQueryClient with explicit credentials
            this.client = BigQueryClient.Create(projectId, credential);

            //client = BigQueryClient.Create(projectId);
            return;
            //try
            //{
            //    try
            //    {
            //        using (var connection = new SqlConnection(stringToConfigureFrom))
            //        {
            //            connection.Open();

            //            //check if table exists
            //            using (var cmd = new SqlCommand())
            //            {
            //                cmd.CommandType = CommandType.Text;
            //                cmd.CommandText = "select top 1 RawFeedId from Traces";
            //                cmd.Connection = connection;
            //                var r = cmd.ExecuteScalar();
            //            }
            //        }
            //        ConnectionStringKey = stringToConfigureFrom;
            //    }
            //    catch (SqlException s)
            //    {
            //        if (s.Message.IndexOf("Invalid object name 'Traces'", StringComparison.OrdinalIgnoreCase) >= 0)
            //        {
            //            try
            //            {
            //                logger.InfoFormat("Start to create 'Traces' table");
            //                using (var connection = new SqlConnection(stringToConfigureFrom))
            //                {
            //                    connection.Open();

            //                    //check if table exists
            //                    using (var cmd = new SqlCommand())
            //                    {
            //                        cmd.CommandType = CommandType.Text;
            //                        cmd.CommandText = GetCreateTableSqlTraces();
            //                        cmd.Connection = connection;
            //                        var r = cmd.ExecuteNonQuery();
            //                    }
            //                }
            //                logger.InfoFormat("'Traces' table created successfully");
            //                Initialize(stringToConfigureFrom);
            //            }
            //            catch (Exception exx)
            //            {
            //                logger.Error("Error on creating 'Traces' table in database. Please check if the sql user has the rights to create table.");
            //                throw exx;
            //            }
            //        }
            //        else
            //        {
            //            logger.Error("Please check the CrackerConfiguration value from config file. Expected a valid connection string with proper rights.");
            //            throw s;
            //        }
            //    }
            //}
            //catch (Exception ex)
            //{
            //    logger.Error("An error occurs on initialize cracker using CrackerConfiguration value from config file. Expected a valid connection string with proper rights.");
            //    throw ex;
            //}
        }

        public void ToApp(Message message)
        {
            logger.DebugFormat("ToApp...{0}", message.ToString());
        }

        public void FromAdmin(Message message)
        {
            logger.DebugFormat("FromAdmin...{0}", message.ToString());
        }

        public void ToAdmin(Message message)
        {
            logger.DebugFormat("ToAdmin...{0}", message.ToString());
        }
    }
}

